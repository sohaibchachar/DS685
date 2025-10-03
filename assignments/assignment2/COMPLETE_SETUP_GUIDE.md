# Complete Setup Guide: Object Detection with Semantic Localization

## Quick Start Commands

### 1. Start TurtleBot World & Gazebo
```bash
cd /workspaces/eng-ai-agents
./run_gazebo_rviz.sh
```

### 2. Start Object Detection (Without Semantic Features)
```bash
# Terminal 1: Start TurtleBot world
cd /workspaces/eng-ai-agents
./run_gazebo_rviz.sh

# Terminal 2: Start object detection (regular)
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection
source /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/install/setup.bash
ros2 run turtlebot_object_detection object_detection_node
```

### 3. Start Object Detection WITH Semantic Features
```bash
# Terminal 1: Start TurtleBot world
cd /workspaces/eng-ai-agents
./run_gazebo_rviz.sh

# Terminal 2: Start PostgreSQL database
sudo service postgresql start

# Terminal 3: Start semantic object detection
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection
source /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/install/setup.bash

# Set environment variables
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=agents
export PGUSER=postgres
export PGPASSWORD=postgres
export SEMANTIC_ENABLED=true

# Start semantic detection
ros2 run turtlebot_object_detection object_detection_node
```

---

## Fresh Setup Instructions (After PC Restart)

### Step 1: Start Docker Container
```bash
# Navigate to project directory
cd /path/to/your/project

# Start the Docker environment (run whatever command you use to enter the container)
# This depends on your specific Docker setup
```

### Step 2: Install PostgreSQL + pgvector (One-time Setup)
```bash
# Download the setup script
cd /workspaces/eng-ai-agents/assignments/assignment2
chmod +x setup_postgres_simple.sh

# Run the setup script
./setup_postgres_simple.sh

# If the script fails, run manually:
sudo apt update
sudo apt install -y postgresql-16 postgresql-contrib libpq-dev postgresql-server-dev-16 build-essential git python3-psycopg2 python3-yaml

# Start PostgreSQL
sudo service postgresql start

# Create database and user
sudo -n su postgres -c "psql -c \"CREATE DATABASE agents;\""
sudo -n su postgres -c "psql -c \"ALTER USER postgres WITH SUPERUSER PASSWORD 'postgres';\""
sudo -n su postgres -c "psql -c \"GRANT ALL PRIVILEGES ON DATABASE agents TO postgres;\""

# Build and install pgvector
cd /tmp
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Enable vector extension
sudo -n su postgres -c "psql -d agents -c \"CREATE EXTENSION vector;\""
```

### Step 3: Install Python Dependencies
```bash
# Install required Python packages
pip install transformers psycopg2-binary pyyaml shapely

# Or if using system packages:
sudo apt install -y python3-psycopg2 python3-yaml python3-shapely
pip install transformers shapely --break-system-packages
```

### Step 4: Rebuild ROS Package
```bash
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection
rm -rf build/ install/ log/
colcon build --packages-select turtlebot_object_detection
```

### Step 5: Verify Database Setup
```bash
# Test database connection
python3 -c "
import psycopg2
conn = psycopg2.connect(host='localhost', port='5432', database='agents', user='postgres', password='postgres')
cur = conn.cursor()
cur.execute('SELECT extversion FROM pg_extension WHERE extname = \\'vector\\';')
print('✅ Database ready:', cur.fetchone()[0])
conn.close()
"
```

---

## Starting Everything (Fresh Session)

### Terminal 1: Start TurtleBot Environment
```bash
cd /workspaces/eng-ai-agents
./run_gazebo_rviz.sh
```

### Terminal 2: Start PostgreSQL
```bash
sudo service postgresql start
```

### Terminal 3: Start Semantic Object Detection
```bash
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection
source /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/install/setup.bash

# Set database environment variables
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=agents
export PGUSER=postgres
export PGPASSWORD=postgres
export SEMANTIC_ENABLED=true

# Start detection node
ros2 run turtlebot_object_detection object_detection_node
```

---

## Navigation Commands

### Moving TurtleBot to See Objects
```bash
# Open Terminal 4
cd /workspaces/eng-ai-agents
source /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/install/setup.bash

# Navigate to platform 1 (bench area)
ros2 run turtlebot3_navigation2 navigate_action_client.py --ros-args --params /opt/ros/jazzy/share/turtlebot3_navigation2/param/waffle.yaml &

# Or use manual navigation
ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.5, y: 0.0, z: 0.0}, angular: {z: 0.0}}' --once
```

---

## Querying Semantic Database

### Basic Queries
```bash
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/turtlebot_object_detection

# Query objects by name
python3 query_semantic_db.py "banana"
python3 query_semantic_db.py "apple"
python3 query_semantic_db.py "person"

# Query by category
python3 query_semantic_db.py "fruit"
python3 query_semantic_db.py "furniture"
```

### Direct Database Access
```bash
# Connect to database directly
sudo -u postgres psql -d agents

# Inside psql:
# List all observations
SELECT label, robot_x, robot_y, created_at FROM object_observations ORDER BY created_at DESC LIMIT 10;

# Count objects by type
SELECT label, COUNT(*) FROM object_observations GROUP BY label ORDER BY COUNT(*) DESC;

# View regions
SELECT * FROM regions;

# Exit psql
\q
```

---

## Troubleshooting

### PostgreSQL Issues
```bash
# Check PostgreSQL status
sudo service postgresql status

# Restart PostgreSQL
sudo service postgresql restart

# Check if pgvector extension is loaded
sudo -u postgres psql -d agents -c "\dx"

# Check database connection
psql -h localhost -p 5432 -U postgres -d agents -c "SELECT 1;"
```

### Python/ROS Package Issues
```bash
# Rebuild ROS package
cd /workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection
rm -rf build/ install/ log/
colcon build --packages-select turtlebot_object_detection

# Check if semantic module imports correctly
python3 -c "from turtlebot_object_detection.semantic_localization import SemanticLocalizer; print('✅ Semantic module OK')"

# Check ROS package installation
ros2 pkg list | grep turtlebot_object_detection
```

### Environment Variables
```bash
# Check current environment
echo "SEMANTIC_ENABLED: $SEMANTIC_ENABLED"
echo "PGHOST: $PGHOST"
echo "PGDATABASE: $PGDATABASE"

# Reset environment if needed
export PGHOST=localhost
export PGPORT=5432
export PGDATABASE=agents
export PGUSER=postgres
export PGPASSWORD=postgres
export SEMANTIC_ENABLED=true
```

---

## File Locations

### Main Files
- **Detection Node**: `/workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/turtlebot_object_detection/object_detection_node.py`
- **Semantic Module**: `/workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/turtlebot_object_detection/semantic_localization.py`
- **Database Adapter**: `/workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/turtlebot_object_detection/database_adapter.py`
- **Query Script**: `/workspaces/eng-ai-agents/assignments/assignment2/turtlebot_object_detection/turtlebot_object_detection/query_semantic_db.py`

### Setup Scripts
- **PostgreSQL Setup**: `/workspaces/eng-ai-agents/assignments/assignment2/setup_postgres_simple.sh`
- **Gazebo Launcher**: `/workspaces/eng-ai-agents/run_gazebo_rviz.sh`

### Database Location
- **Database**: `agents` (PostgreSQL)
- **Data Directory**: `/var/lib/postgresql/16/main/`

---

## Environment Summary

### Without Semantic Features (Default)
- Runs exactly like original object detection
- No database required
- No environment variables needed
- Same performance and functionality

### With Semantic Features
- **Database**: PostgreSQL-16 + pgvector
- **Storage**: Vector embeddings, robot poses, region mapping
- **Smart Filtering**: Deduplication, movement thresholds
- **Query Interface**: Semantic text-based object lookup

### Key Differences
- **Processing**: Each detection gets cropped and embedded
- **Storage**: Vector database stores semantic representations
- **Output**: Same visual annotations + semantic database records
- **Query**: Can ask "Where are bananas?" and get bench locations
