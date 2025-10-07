#!/usr/bin/env python3
"""
Simple PostgreSQL + pgvector storage for semantic localization.

Tables:
- regions(id, name UNIQUE, kind, created_at)
- objects(id, class_name, class_id, confidence, embedding vector(2048), created_at)
- object_observations(id, object_id, region_id, robot_x, robot_y, robot_theta, bbox_json, created_at)

Usage:
  db = SemanticDB()
  obj_id = db.insert_object(class_name, class_id, confidence, embedding)
  db.insert_observation(obj_id, robot_x, robot_y, robot_theta, bbox, region_name)
"""

import os
import json
import psycopg2
import psycopg2.extras
from typing import List, Dict, Optional, Tuple


class SemanticDB:
    def __init__(self) -> None:
        # Default database configuration - no need for environment variables
        self.config = {
            'host': os.getenv('POSTGRES_HOST', 'localhost'),
            'port': int(os.getenv('POSTGRES_PORT', '5432')),
            'database': os.getenv('POSTGRES_DB', 'agents'),
            'user': os.getenv('POSTGRES_USER', 'postgres'),
            'password': os.getenv('POSTGRES_PASSWORD', 'postgres'),
        }
        self.conn = None
        self.connect()
        self.ensure_schema()

    def connect(self) -> None:
        try:
            # Try to connect with the configured settings
            self.conn = psycopg2.connect(**self.config)
            print(f"✅ Connected to database: {self.config['database']} on {self.config['host']}:{self.config['port']}")
        except psycopg2.OperationalError as e:
            if "password authentication failed" in str(e).lower():
                print(f"❌ Database password authentication failed")
                print(f"💡 Trying with default password...")
                # Try with default password
                self.config['password'] = 'postgres'
                self.conn = psycopg2.connect(**self.config)
                print(f"✅ Connected to database with default password")
            else:
                print(f"❌ Failed to connect to database: {e}")
                raise
        except Exception as e:
            print(f"❌ Database connection error: {e}")
            raise

    def ensure_schema(self) -> None:
        cur = self.conn.cursor()
        # pgvector
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        # regions
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS regions (
                id SERIAL PRIMARY KEY,
                name VARCHAR(128) UNIQUE NOT NULL,
                kind VARCHAR(64) DEFAULT 'unknown',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # objects (2048-d ResNet50 embedding)
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS objects (
                id SERIAL PRIMARY KEY,
                class_name VARCHAR(128) NOT NULL,
                class_id INT NOT NULL,
                confidence FLOAT NOT NULL,
                embedding vector(2048),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # observations
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS object_observations (
                id SERIAL PRIMARY KEY,
                object_id INT REFERENCES objects(id) ON DELETE CASCADE,
                region_id INT REFERENCES regions(id) ON DELETE SET NULL,
                robot_x FLOAT NOT NULL,
                robot_y FLOAT NOT NULL,
                robot_theta FLOAT NOT NULL,
                bbox_json JSONB NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )

        # indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_objects_class ON objects(class_name);")
        # Skip ivfflat index for large vectors (2048-dim), use regular index instead
        cur.execute("CREATE INDEX IF NOT EXISTS idx_objects_embed_basic ON objects(class_name, confidence);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_region ON object_observations(region_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_obs_created ON object_observations(created_at DESC);")

        self.conn.commit()
        cur.close()

    @staticmethod
    def _to_pgvector(values) -> str:
        return '[' + ','.join(str(float(v)) for v in values) + ']'

    def upsert_region(self, name: str, kind: str = 'unknown') -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO regions (name, kind) VALUES (%s, %s)
            ON CONFLICT (name) DO UPDATE SET kind = EXCLUDED.kind
            RETURNING id
            """,
            (name, kind),
        )
        rid = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return rid

    def insert_object(self, class_name: str, class_id: int, confidence: float, embedding) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO objects (class_name, class_id, confidence, embedding)
            VALUES (%s, %s, %s, %s)
            RETURNING id
            """,
            (class_name, class_id, confidence, self._to_pgvector(embedding)),
        )
        oid = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return oid

    def insert_observation(
        self,
        object_id: int,
        robot_x: float,
        robot_y: float,
        robot_theta: float,
        bbox: List[int],
        region_name: Optional[str] = None,
        region_kind: str = 'unknown_area',
    ) -> int:
        region_id = None
        if region_name:
            region_id = self.upsert_region(region_name, region_kind)
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO object_observations (object_id, region_id, robot_x, robot_y, robot_theta, bbox_json)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
            """,
            (object_id, region_id, robot_x, robot_y, robot_theta, json.dumps(list(map(int, bbox)))),
        )
        obs_id = cur.fetchone()[0]
        self.conn.commit()
        cur.close()
        return obs_id

    def query_locations_for_class(self, class_name: str) -> List[Dict]:
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT r.name AS region_name, r.kind AS region_kind, COUNT(*) AS count
            FROM object_observations obs
            JOIN objects o ON obs.object_id = o.id
            LEFT JOIN regions r ON obs.region_id = r.id
            WHERE o.class_name ILIKE %s
            GROUP BY r.name, r.kind
            ORDER BY count DESC
            """,
            (class_name,),
        )
        rows = cur.fetchall()
        cur.close()
        return [dict(r) for r in rows]

    def knn_query(self, embedding, top_k: int = 5) -> List[Dict]:
        cur = self.conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(
            """
            SELECT o.id, o.class_name, o.confidence, 1 - (o.embedding <=> %s) AS similarity
            FROM objects o
            ORDER BY o.embedding <=> %s
            LIMIT %s
            """,
            (self._to_pgvector(embedding), self._to_pgvector(embedding), top_k),
        )
        rows = cur.fetchall()
        cur.close()
        return [dict(r) for r in rows]


