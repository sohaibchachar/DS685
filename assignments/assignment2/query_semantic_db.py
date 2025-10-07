#!/usr/bin/env python3
"""
Simple database query interface for semantic localization data
Run this script to explore stored detections and semantic locations
"""

import sys
import os

# Add the package path so we can import semantic_db
sys.path.append(os.path.join(os.path.dirname(__file__), 'turtlebot_object_detection'))

try:
    # Try importing as module first
    from turtlebot_object_detection.semantic_db import SemanticDB
except ImportError:
    try:
        # Fallback to direct import
        from semantic_db import SemanticDB
    except ImportError:
        print("❌ Could not import semantic_db. Make sure PostgreSQL is set up and running.")
        sys.exit(1)


def main():
    """Main query interface"""
    try:
        db = SemanticDB()
        print("✅ Connected to semantic database")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        print("💡 Make sure PostgreSQL is running: bash setup_postgres_simple.sh")
        return
    
    print("\n" + "="*60)
    print("🔍 SEMANTIC DATABASE EXPLORER")
    print("="*60)
    
    # Show statistics
    print(f"\n📊 QUICK STATS:")
    
    # Count objects
    db.conn.commit()  # Ensure fresh data
    cur = db.conn.cursor()
    cur.execute("SELECT COUNT(*) FROM objects")
    object_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM object_observations") 
    observation_count = cur.fetchone()[0]
    
    cur.execute("SELECT COUNT(*) FROM regions")
    region_count = cur.fetchone()[0]
    
    print(f"   Objects detected: {object_count}")
    print(f"   Total observations: {observation_count}")
    print(f"   Regions visited: {region_count}")
    
    # Recent detections
    print(f"\n🔥 RECENT DETECTIONS:")
    cur.execute("""
        SELECT o.class_name, COALESCE(r.name, 'unknown') AS region, 
               obs.created_at
        FROM object_observations obs
        JOIN objects o ON obs.object_id = o.id
        LEFT JOIN regions r ON obs.region_id = r.id
        ORDER BY obs.created_at DESC
        LIMIT 10
    """)
    
    recent = cur.fetchall()
    if recent:
        for class_name, region, timestamp in recent:
            print(f"   • {class_name} at {region} ({timestamp})")
    else:
        print("   No detections yet")
    
    # Objects by region
    print(f"\n📍 OBJECTS BY REGION:")
    cur.execute("""
        SELECT COALESCE(r.name, 'unknown') AS region, 
               o.class_name, COUNT(*) as count
        FROM object_observations obs
        JOIN objects o ON obs.object_id = o.id
        LEFT JOIN regions r ON obs.region_id = r.id
        GROUP BY r.name, o.class_name
        ORDER BY region, count DESC
    """)
    
    regions = cur.fetchall()
    if regions:
        current_region = None
        for region, class_name, count in regions:
            if region != current_region:
                print(f"\n   🏷️  {region}:")
                current_region = region
            print(f"      • {class_name}: {count}")
    else:
        print("   No location data yet")
    
    cur.close()
    
    print(f"\n💡 MORE QUERY EXAMPLES:")
    print("   # Find all cakes:")
    print("   python3 -c \"from semantic_db import SemanticDB; print(SemanticDB().query_locations_for_class('cake'))\"")
    print("   ")
    print("   # Direct SQL:")
    print("   PGPASSWORD=postgres psql -h localhost -p 5432 -U postgres -d agents")


if __name__ == '__main__':
    main()
