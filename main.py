"""
Main application entry point for TradingView Strategy Generator
Handles command-line interface and application startup.
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from config.config_manager import ConfigManager
from core.database import DatabaseManager
from modules.pinescript_converter import PineScriptConverter
from modules.backtest_engine import BacktestEngine, SimpleMovingAverageStrategy
from modules.optimizer import StrategyOptimizer, ParameterSpace

def setup_logging(log_level: str = "INFO"):
    """Setup application logging."""
    # Create logs directory if it doesn't exist
    Path("logs").mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

def run_web_app():
    """Launch the Streamlit web application."""
    import subprocess
    
    print("🚀 Starting TradingView Strategy Generator Web App...")
    print("📊 Open your browser to: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            "streamlit", "run", "src/ui/main_app.py",
            "--server.port=8501",
            "--server.address=localhost"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down server...")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting web app: {e}")
        sys.exit(1)

def run_cli_demo():
    """Run a command-line demonstration of the application."""
    print("🔧 TradingView Strategy Generator - CLI Demo")
    print("=" * 50)
    
    # Initialize components
    print("📋 Initializing components...")
    config = ConfigManager()
    db = DatabaseManager()
    converter = PineScriptConverter()
    backtest_engine = BacktestEngine(config.get_backtesting_config())
    optimizer = StrategyOptimizer(backtest_engine, config.get_optimization_config())
    
    print("✅ Components initialized successfully!")
    
    # Demo Pine Script conversion
    print("\n🔄 Demo: Converting Pine Script to Python")
    print("-" * 40)
    
    sample_pinescript = '''
//@version=5
strategy("Demo MA Crossover", overlay=true)

fast_length = input(10, "Fast MA Length")
slow_length = input(30, "Slow MA Length")

fast_ma = ta.sma(close, fast_length)
slow_ma = ta.sma(close, slow_length)

long_condition = ta.crossover(fast_ma, slow_ma)
short_condition = ta.crossunder(fast_ma, slow_ma)

if long_condition
    strategy.entry("Long", strategy.long)

if short_condition
    strategy.entry("Short", strategy.short)
'''
    
    print("Pine Script Input:")
    print(sample_pinescript)
    
    try:
        python_code, warnings = converter.convert(sample_pinescript)
        print("\n✅ Conversion successful!")
        if warnings:
            print(f"⚠️  {len(warnings)} warnings generated")
        
        # Save the converted strategy
        strategy = db.save_strategy(
            name="Demo MA Crossover Strategy",
            code=python_code,
            strategy_type="python",
            description="Demo strategy converted from Pine Script"
        )
        print(f"💾 Strategy saved with ID: {strategy.id}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return
    
    # Demo backtesting
    print("\n🔬 Demo: Running Backtest")
    print("-" * 30)
    
    try:
        print("📊 Running backtest on demo strategy...")
        results = backtest_engine.quick_backtest(SimpleMovingAverageStrategy, "BTC-USD")
        
        if results['success']:
            print("✅ Backtest completed successfully!")
            print(f"💰 Net Profit: ${results['net_profit']:.2f}")
            print(f"📈 Profit Factor: {results['profit_factor']:.2f}")
            print(f"📉 Max Drawdown: {results['max_drawdown']:.2%}")
            print(f"🎯 Win Rate: {results['win_rate']:.2%}")
            print(f"🔢 Number of Trades: {results['number_of_trades']}")
            print(f"⏱️  Execution Time: {results['execution_time']:.2f}s")
        else:
            print(f"❌ Backtest failed: {results.get('error')}")
    
    except Exception as e:
        print(f"❌ Backtest error: {e}")
    
    # Demo optimization
    print("\n⚡ Demo: Parameter Optimization")
    print("-" * 35)
    
    try:
        print("🔍 Setting up parameter space...")
        param_space = ParameterSpace()
        param_space.add_integer('fast_period', 5, 15)
        param_space.add_integer('slow_period', 20, 40)
        
        print("🚀 Running optimization (this may take a moment)...")
        opt_results = optimizer.optimize_strategy(
            SimpleMovingAverageStrategy,
            'BTC-USD',
            param_space,
            algorithm='random_search'
        )
        
        print("✅ Optimization completed!")
        print(f"🏆 Best Parameters: {opt_results.best_params}")
        print(f"📊 Best Score: {opt_results.best_score:.4f}")
        print(f"🔄 Iterations: {opt_results.iterations}")
        print(f"⏱️  Execution Time: {opt_results.execution_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Optimization error: {e}")
    
    # Show database stats
    print("\n📊 Database Statistics")
    print("-" * 25)
    
    strategies = db.get_all_strategies()
    open_signals = db.get_open_signals()
    
    print(f"📈 Total Strategies: {len(strategies)}")
    print(f"📡 Open Signals: {len(open_signals)}")
    
    print("\n🎉 Demo completed successfully!")
    print("🌐 To use the web interface, run: python main.py --web")

def run_tests():
    """Run the test suite."""
    print("🧪 Running TradingView Strategy Generator Test Suite")
    print("=" * 55)
    
    try:
        import pytest
        
        # Run tests with verbose output
        exit_code = pytest.main([
            "src/tests/test_suite.py",
            "-v",
            "--tb=short",
            "--color=yes"
        ])
        
        if exit_code == 0:
            print("\n✅ All tests passed!")
        else:
            print(f"\n❌ Tests failed with exit code: {exit_code}")
            
        return exit_code
        
    except ImportError:
        print("❌ pytest not installed. Run: pip install pytest")
        return 1
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return 1

def show_system_info():
    """Display system information and requirements."""
    print("📋 TradingView Strategy Generator - System Information")
    print("=" * 60)
    
    # Python version
    print(f"🐍 Python Version: {sys.version}")
    
    # Check required packages
    required_packages = [
        "streamlit", "pandas", "numpy", "yfinance", "backtrader",
        "sqlalchemy", "plotly", "ta", "scikit-optimize", "pyyaml"
    ]
    
    print("\n📦 Package Status:")
    print("-" * 20)
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - NOT INSTALLED")
    
    # Configuration status
    try:
        config = ConfigManager()
        print(f"\n⚙️  Configuration: ✅ Loaded")
        print(f"📊 Data Sources: Yahoo Finance ({'✅' if config.get('data.yahoo_finance.enabled') else '❌'})")
        print(f"🤖 Telegram Bot: {'✅' if config.is_telegram_enabled() else '❌'}")
        print(f"📡 Live Signals: {'✅' if config.is_signals_enabled() else '❌'}")
    except Exception as e:
        print(f"\n⚙️  Configuration: ❌ Error loading - {e}")
    
    # Database status
    try:
        db = DatabaseManager()
        strategies = db.get_all_strategies()
        print(f"\n💾 Database: ✅ Connected ({len(strategies)} strategies)")
    except Exception as e:
        print(f"\n💾 Database: ❌ Error - {e}")

def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(
        description="TradingView Strategy Generator - Convert Pine Script to Python and optimize trading strategies"
    )
    
    parser.add_argument(
        "--web", 
        action="store_true", 
        help="Launch the web interface (default)"
    )
    
    parser.add_argument(
        "--demo", 
        action="store_true", 
        help="Run CLI demo showing all features"
    )
    
    parser.add_argument(
        "--test", 
        action="store_true", 
        help="Run the test suite"
    )
    
    parser.add_argument(
        "--info", 
        action="store_true", 
        help="Show system information and requirements"
    )
    
    parser.add_argument(
        "--log-level", 
        choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
        default="INFO",
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Handle command line options
    if args.test:
        exit_code = run_tests()
        sys.exit(exit_code)
    elif args.demo:
        run_cli_demo()
    elif args.info:
        show_system_info()
    else:
        # Default to web interface
        run_web_app()

if __name__ == "__main__":
    main()