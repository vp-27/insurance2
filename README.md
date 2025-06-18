# 🏢 Live Insurance Risk & Quote Co-Pilot

A real-time FinTech application built with Pathway that provides live risk assessment and dynamic insurance pricing for commercial properties. This hackathon project demonstrates how AI can modernize insurance underwriting by processing live data streams and providing instant risk analysis.

## 🧠 Problem Solved

Traditional insurance underwriting relies on stale data and manual processes. Live incidents like weather alerts, fires, floods, or crime can materially affect property risk profiles, but quotes often lag behind reality. This system addresses that gap by providing:

- **Live Risk Summaries**: Real-time analysis of current risk factors
- **Dynamic Risk Scores**: 1-10 scale updated as new data arrives  
- **Instant Quote Updates**: Insurance premiums that reflect current conditions

## 🚀 Architecture & Features

### Core Components

1. **Data Fetcher** (`data_fetcher.py`)
   - Fetches live weather alerts from National Weather Service API
   - Monitors local news for incidents (fire, flood, crime)
   - Simulates additional risk data sources
   - Saves timestamped alerts as JSON files

2. **Pathway RAG Pipeline** (`pipeline.py`)
   - Streams JSON files using Pathway's real-time processing
   - Generates vector embeddings using sentence-transformers
   - Maintains live-updating vector index
   - Integrates with OpenRouter-compatible LLMs for analysis

3. **FastAPI Web Application** (`app.py`)
   - React-based frontend with real-time updates
   - RESTful API for risk assessments
   - Demo controls for testing different scenarios
   - Auto-refresh capability for live monitoring

### Key Features

✅ **Real-time Data Processing**: Pathway streaming engine processes new alerts instantly  
✅ **Dynamic Vector Index**: No manual reloads - embeddings update automatically  
✅ **LLM-Powered Analysis**: Uses Mixtral/LLaMA for intelligent risk assessment  
✅ **Live UI Updates**: Frontend refreshes every 30 seconds for real-time effect  
✅ **Demo Mode**: Inject test alerts to see immediate risk/quote changes  

## 📦 Installation & Setup

### Prerequisites

- Python 3.8+
- API Keys (optional, has fallback simulation modes):
  - OpenRouter API key for LLM access
  - NewsData.io API key for news feeds

### Quick Start

1. **Clone and Install Dependencies**
```bash
git clone <repository>
cd insurance2
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys (optional - system works without them)
```

3. **Run the Application**
```bash
# Start the data fetcher (in one terminal)
python data_fetcher.py

# Start the main application (in another terminal) 
python app.py
```

4. **Access the Web Interface**
Open http://localhost:8000 in your browser

## 🎯 Usage & Demo

### Basic Usage

1. Enter a property address (e.g., "25 Columbus Dr, Jersey City, NJ")
2. Click "Get Live Assessment" 
3. View risk score (1-10) and monthly insurance quote
4. Enable auto-refresh for live monitoring

### Demo Scenarios

Use the demo controls to inject test alerts and see real-time updates:

- **🔥 Fire Alert**: Raises risk score to 9/10, quote to ~$950/month
- **🌊 Flood Alert**: Risk score 7/10, quote ~$850/month  
- **🚨 Crime Alert**: Risk score 5/10, quote ~$750/month

### API Endpoints

- `POST /get_assessment`: Get risk assessment for an address
- `POST /inject_test_alert`: Inject test alerts for demo
- `GET /health`: System health check
- `GET /stats`: View system statistics

## 🔧 Technical Details

### Pathway Integration

The system uses Pathway's streaming capabilities to:
- Monitor `./live_data_feed/` directory for new JSON files
- Automatically process and index new documents
- Update vector embeddings without manual intervention
- Provide real-time query responses

### Risk Calculation

```python
base_cost = $500
risk_multiplier = 0.1
quote = base_cost × (1 + risk_multiplier × risk_score)
```

### Data Sources

- **Weather**: National Weather Service API (free)
- **News**: NewsData.io API (with simulation fallback)
- **Crime**: Simulated data (many crime APIs require special access)
- **Manual Injection**: Demo controls for testing

## 📊 Example Workflow

### Initial Assessment
```
Address: "25 Columbus Dr, Jersey City, NJ"
Risk Score: 2/10 (No current alerts)
Quote: $600/month
```

### After Fire Alert Injection
```
Alert Injected: "4-alarm fire reported near 25 Columbus Dr"
Risk Score: 9/10 (Critical operational disruption)  
Quote: $950/month
Updated: Within 30 seconds
```

## 🛠 Development

### Project Structure
```
insurance2/
├── app.py              # FastAPI web application
├── pipeline.py         # Pathway RAG pipeline
├── data_fetcher.py     # Live data collection
├── requirements.txt    # Python dependencies
├── .env               # Configuration
├── live_data_feed/    # JSON alert storage
└── README.md          # This file
```

### Customization

- **Add Data Sources**: Extend `DataFetcher` class with new APIs
- **Modify Risk Logic**: Update LLM prompts in `pipeline.py`
- **UI Changes**: Edit React components in `app.py`
- **Deployment**: Configure for cloud hosting (AWS, GCP, etc.)

## 🔍 Monitoring & Debugging

### Health Checks
```bash
curl http://localhost:8000/health
curl http://localhost:8000/stats
```

### Log Files
- Check console output for data fetching status
- Monitor `live_data_feed/` directory for new alerts
- API responses include document counts and timestamps

### Common Issues

1. **No API Keys**: System works in simulation mode
2. **Missing Dependencies**: Run `pip install -r requirements.txt`
3. **Port Conflicts**: Change port in `app.py` if needed
4. **Data Directory**: Ensure `live_data_feed/` exists and is writable

## 🌟 Hackathon Demo Points

- **Real-time Processing**: Show alerts appearing in data directory
- **Live UI Updates**: Demonstrate auto-refresh functionality  
- **Risk Impact**: Inject fire alert and show quote increase
- **No Manual Refresh**: Vector index updates automatically
- **Production Ready**: FastAPI + React frontend

## 🔮 Future Enhancements

- **Geographic Expansion**: Support multiple cities/regions
- **Historical Analysis**: Trend analysis and predictive modeling
- **Integration APIs**: Connect with actual insurance systems
- **Mobile App**: React Native frontend
- **Advanced Alerts**: IoT sensors, satellite imagery, social media

## 📄 License

MIT License - see LICENSE file for details

---

Built with ❤️ using Pathway, FastAPI, React, and OpenRouter AI
