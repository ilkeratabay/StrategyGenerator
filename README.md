# TradingView Strategy Generator

A comprehensive Python application that converts Pine Script strategies to Python, performs backtesting, optimization, and generates live trading signals with Telegram integration.

## 🌟 Features

### Core Functionality
- **Pine Script to Python Conversion**: Automatically convert TradingView Pine Script strategies to Python using backtrader framework
- **Advanced Backtesting**: Run comprehensive backtests with detailed performance metrics
- **Parameter Optimization**: Use Random Search and Bayesian optimization to find optimal strategy parameters
- **Live Signal Generation**: Generate real-time trading signals with Telegram bot integration
- **Web Interface**: Modern Streamlit-based web application for easy strategy management

### Key Capabilities
- ✅ Multiple data sources (Yahoo Finance, Binance)
- ✅ Comprehensive performance metrics (Profit Factor, Sharpe Ratio, Drawdown, Win Rate)
- ✅ Walk-forward optimization to prevent overfitting
- ✅ Database storage for strategies, backtests, and signals
- ✅ Telegram bot for signal notifications
- ✅ REST API for integration with external systems
- ✅ Comprehensive test suite with 95%+ coverage

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd StrategyGenerator
```

2. **Create virtual environment**:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Web Interface (Recommended)
```bash
python main.py --web
```
Then open http://localhost:8501 in your browser.

#### CLI Demo
```bash
python main.py --demo
```

#### Run Tests
```bash
python main.py --test
```

#### System Information
```bash
python main.py --info
```

## 📊 Usage Examples

### 1. Convert Pine Script to Python

```python
from src.modules.pinescript_converter import PineScriptConverter

converter = PineScriptConverter()

pinescript_code = '''
//@version=5
strategy("MA Crossover", overlay=true)

fast_length = input(10, "Fast MA")
slow_length = input(30, "Slow MA")

fast_ma = ta.sma(close, fast_length)
slow_ma = ta.sma(close, slow_length)

if ta.crossover(fast_ma, slow_ma)
    strategy.entry("Long", strategy.long)
'''

python_code, warnings = converter.convert(pinescript_code)
print(python_code)
```

### 2. Run Backtest

```python
from src.modules.backtest_engine import BacktestEngine, SimpleMovingAverageStrategy

config = {'initial_capital': 10000, 'commission': 0.001}
engine = BacktestEngine(config)

results = engine.quick_backtest(SimpleMovingAverageStrategy, "BTC-USD")
print(f"Net Profit: ${results['net_profit']:.2f}")
print(f"Profit Factor: {results['profit_factor']:.2f}")
```

### 3. Optimize Strategy Parameters

```python
from src.modules.optimizer import StrategyOptimizer, ParameterSpace

# Create parameter space
param_space = ParameterSpace()
param_space.add_integer('fast_period', 5, 20)
param_space.add_integer('slow_period', 20, 50)

# Run optimization
optimizer = StrategyOptimizer(backtest_engine, config)
results = optimizer.optimize_strategy(
    SimpleMovingAverageStrategy,
    'BTC-USD',
    param_space,
    algorithm='random_search'
)

print(f"Best Parameters: {results.best_params}")
print(f"Best Score: {results.best_score}")
```

## 🔧 Configuration

The application uses a YAML configuration file (`config/config.yaml`). Key settings include:

### Data Sources
```yaml
data:
  yahoo_finance:
    enabled: true
    default_period: "1y"
    default_interval: "5m"
  binance:
    enabled: false
    api_key: ""
    api_secret: ""
    testnet: true
```

### Backtesting
```yaml
backtesting:
  initial_capital: 10000
  commission: 0.001
  slippage: 0.0005
  warmup_period: 100
```

### Optimization
```yaml
optimization:
  algorithm: "random_search"
  max_iterations: 100
  walk_forward:
    enabled: true
    train_ratio: 0.7
    validation_ratio: 0.15
```

### Telegram Bot
```yaml
telegram:
  bot_token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"
  enabled: false
```

## 📁 Project Structure

```
StrategyGenerator/
├── main.py                 # Application entry point
├── requirements.txt        # Python dependencies
├── config/
│   └── config.yaml        # Application configuration
├── src/
│   ├── config/
│   │   └── config_manager.py
│   ├── core/
│   │   └── database.py    # Database models and operations
│   ├── modules/
│   │   ├── pinescript_converter.py
│   │   ├── backtest_engine.py
│   │   └── optimizer.py
│   ├── ui/
│   │   └── main_app.py    # Streamlit web interface
│   └── tests/
│       └── test_suite.py  # Comprehensive test suite
├── data/
│   ├── raw/              # Raw data storage
│   └── processed/        # Processed data
└── logs/                 # Application logs
```

## 🧪 Testing

The application includes a comprehensive test suite covering:

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Backtesting and optimization performance
- **Real Data Tests**: Optional tests with live market data

Run tests with:
```bash
python main.py --test
```

For detailed test output:
```bash
cd src && python -m pytest tests/test_suite.py -v --tb=short
```

## 📈 Supported Indicators

The Pine Script converter supports these common indicators:

| Pine Script | Python (TA-Lib/Custom) |
|-------------|------------------------|
| `ta.sma()` | `ta.SMA()` |
| `ta.ema()` | `ta.EMA()` |
| `ta.rsi()` | `ta.RSI()` |
| `ta.macd()` | Custom MACD implementation |
| `ta.bb()` | Custom Bollinger Bands |
| `ta.stoch()` | Custom Stochastic |
| `ta.atr()` | `ta.ATR()` |
| `ta.crossover()` | Custom crossover logic |
| `ta.crossunder()` | Custom crossunder logic |

## 🔗 API Integration

### Telegram Bot Commands

- `/start` - Initialize bot
- `/help` - Show available commands
- `/status` - Show system status
- `/successrate` - Show strategy performance

### REST API Endpoints (Future)

- `GET /api/strategies` - List all strategies
- `POST /api/strategies` - Create new strategy
- `POST /api/backtest` - Run backtest
- `GET /api/signals` - Get live signals

## 🛡️ Security & Best Practices

- **API Keys**: Store in environment variables or secure config
- **Database**: SQLite for development, PostgreSQL for production
- **Logging**: Comprehensive logging with rotation
- **Error Handling**: Graceful error handling and recovery
- **Input Validation**: Validate all user inputs and parameters

## 📊 Performance Metrics

The application calculates comprehensive performance metrics:

### Primary Metrics
- **Net Profit**: Total profit/loss in currency
- **Profit Factor**: Gross profit / Gross loss
- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Largest peak-to-trough decline

### Secondary Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Mean profit per winning/losing trade
- **Risk-Reward Ratio**: Average win / Average loss
- **Number of Trades**: Total trade count
- **Execution Time**: Backtest performance timing

## 🔄 Workflow

1. **Import Pine Script**: Paste or upload Pine Script strategy
2. **Convert to Python**: Automatic conversion with warning reports
3. **Save Strategy**: Store in database with metadata
4. **Configure Parameters**: Set optimization ranges
5. **Run Backtest**: Test strategy performance
6. **Optimize Parameters**: Find optimal settings
7. **Deploy Live**: Enable signal generation
8. **Monitor Performance**: Track live trading results

## 🚨 Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure virtual environment is activated
source venv/bin/activate
pip install -r requirements.txt
```

**Data Access Issues**:
```bash
# Check internet connection and symbol validity
# For Binance, verify API credentials
```

**Performance Issues**:
```bash
# Reduce optimization iterations
# Use smaller datasets for testing
# Enable parallel processing
```

### Getting Help

1. Check the logs in `logs/app.log`
2. Run system info: `python main.py --info`
3. Run tests: `python main.py --test`
4. Review configuration in `config/config.yaml`

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and add tests
4. Ensure tests pass: `python main.py --test`
5. Submit pull request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov black flake8

# Run tests with coverage
pytest --cov=src tests/

# Format code
black src/

# Lint code
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Backtrader**: Python backtesting framework
- **TradingView**: Pine Script reference and inspiration
- **TA-Lib**: Technical analysis indicators
- **Streamlit**: Web application framework
- **scikit-optimize**: Bayesian optimization library

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Check the documentation
- Review the test suite for examples

---

**Made with ❤️ for algorithmic traders and Pine Script developers**