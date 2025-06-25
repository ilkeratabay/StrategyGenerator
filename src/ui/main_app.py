"""
Streamlit Web Application for TradingView Strategy Generator
Main UI interface for strategy management, backtesting, and optimization.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import logging
from typing import Dict, Any, List, Optional
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config_manager import ConfigManager
from core.database import DatabaseManager, Strategy, Backtest, Signal
from modules.pinescript_converter import PineScriptConverter, ConversionError
from modules.backtest_engine import BacktestEngine, SimpleMovingAverageStrategy
from modules.optimizer import StrategyOptimizer, ParameterSpace

# Configure page
st.set_page_config(
    page_title="TradingView Strategy Generator",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'config' not in st.session_state:
    st.session_state.config = ConfigManager()
    st.session_state.config.reload_config()  # Force reload

if 'current_page' not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"

# Force clear cache when config changes
if st.session_state.get('force_reload', False):
    st.session_state.config.reload_config()
    st.session_state.force_reload = False

if 'db' not in st.session_state:
    db_config = st.session_state.config.get_database_config()
    # Construct proper SQLAlchemy URL
    db_type = db_config.get('type', 'sqlite')
    db_path = db_config.get('path', './data/strategies.db')
    
    if db_type == 'sqlite':
        # Ensure the path is absolute for SQLAlchemy
        if not db_path.startswith('sqlite://'):
            if not os.path.isabs(db_path):
                db_path = os.path.abspath(db_path)
            db_url = f'sqlite:///{db_path}'
        else:
            db_url = db_path
    else:
        db_url = f'{db_type}:///{db_path}'
    
    st.session_state.db = DatabaseManager(db_url)

if 'converter' not in st.session_state:
    st.session_state.converter = PineScriptConverter()

if 'backtest_engine' not in st.session_state:
    bt_config = st.session_state.config.get_backtesting_config()
    st.session_state.backtest_engine = BacktestEngine(bt_config)

if 'optimizer' not in st.session_state:
    opt_config = st.session_state.config.get_optimization_config()
    st.session_state.optimizer = StrategyOptimizer(st.session_state.backtest_engine, opt_config)

def main():
    """Main application function."""
    st.title("📈 TradingView Strategy Generator")
    st.markdown("---")
    
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        page = st.selectbox(
            "Choose a page:",
            [
                "🏠 Dashboard",
                "🔄 Pine Script Converter", 
                "📊 Strategy Manager",
                "🔬 Backtesting",
                "⚡ Optimization",
                "📡 Live Signals",
                "⚙️ Settings"
            ],
            index=[
                "🏠 Dashboard",
                "🔄 Pine Script Converter", 
                "📊 Strategy Manager",
                "🔬 Backtesting",
                "⚡ Optimization",
                "📡 Live Signals",
                "⚙️ Settings"
            ].index(st.session_state.current_page)
        )
    
    # Route to selected page
    if page == "🏠 Dashboard":
        st.session_state.current_page = "🏠 Dashboard"
        show_dashboard()
    elif page == "🔄 Pine Script Converter":
        st.session_state.current_page = "🔄 Pine Script Converter"
        show_converter()
    elif page == "📊 Strategy Manager":
        st.session_state.current_page = "📊 Strategy Manager"
        show_strategy_manager()
    elif page == "🔬 Backtesting":
        st.session_state.current_page = "🔬 Backtesting"
        show_backtesting()
    elif page == "⚡ Optimization":
        st.session_state.current_page = "⚡ Optimization"
        show_optimization()
    elif page == "📡 Live Signals":
        st.session_state.current_page = "📡 Live Signals"
        show_live_signals()
    elif page == "⚙️ Settings":
        st.session_state.current_page = "⚙️ Settings"
        show_settings()

def show_dashboard():
    """Display main dashboard."""
    st.header("Dashboard")
    
    # Get recent strategies and backtests
    strategies = st.session_state.db.get_all_strategies()
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Strategies", len(strategies))
    
    with col2:
        # Count total backtests
        session = st.session_state.db.get_session()
        try:
            backtest_count = session.query(Backtest).count()
            st.metric("Total Backtests", backtest_count)
        finally:
            session.close()
    
    with col3:
        # Count open signals
        open_signals = st.session_state.db.get_open_signals()
        st.metric("Open Signals", len(open_signals))
    
    with col4:
        # Show if live signals are enabled
        signals_enabled = st.session_state.config.is_signals_enabled()
        st.metric("Live Signals", "✅ Active" if signals_enabled else "❌ Inactive")
    
    st.markdown("---")
    
    # Recent strategies
    if strategies:
        st.subheader("Recent Strategies")
        
        strategy_data = []
        for strategy in strategies[-5:]:  # Last 5 strategies
            strategy_data.append({
                'Name': strategy.name,
                'Type': strategy.strategy_type,
                'Created': strategy.created_at.strftime('%Y-%m-%d %H:%M'),
                'Active': '✅' if strategy.is_active else '❌'
            })
        
        st.dataframe(pd.DataFrame(strategy_data), use_container_width=True)
    
    # Quick actions
    st.subheader("Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Convert Pine Script", use_container_width=True):
            st.session_state.current_page = "🔄 Pine Script Converter"
            st.rerun()
    
    with col2:
        if st.button("🔬 Run Backtest", use_container_width=True):
            st.session_state.current_page = "🔬 Backtesting"
            st.rerun()
    
    with col3:
        if st.button("⚡ Optimize Strategy", use_container_width=True):
            st.session_state.current_page = "⚡ Optimization"
            st.rerun()

def show_converter():
    """Display Pine Script converter interface."""
    st.header("🔄 Pine Script to Python Converter")
    
    st.markdown("""
    Convert your Pine Script strategies to Python using backtrader framework.
    The converter supports most common Pine Script functions and indicators.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Pine Script Input")
        
        # Example Pine Script
        example_code = '''// Example Pine Script Strategy
//@version=5
strategy("Simple MA Crossover", overlay=true)

// Inputs
fast_length = input(10, "Fast MA Length")
slow_length = input(30, "Slow MA Length")

// Calculate moving averages
fast_ma = ta.sma(close, fast_length)
slow_ma = ta.sma(close, slow_length)

// Plot MAs
plot(fast_ma, color=color.blue, title="Fast MA")
plot(slow_ma, color=color.red, title="Slow MA")

// Entry conditions
long_condition = ta.crossover(fast_ma, slow_ma)
short_condition = ta.crossunder(fast_ma, slow_ma)

// Strategy entries
if long_condition
    strategy.entry("Long", strategy.long)

if short_condition
    strategy.entry("Short", strategy.short)
'''
        
        pinescript_code = st.text_area(
            "Paste your Pine Script code here:",
            value=example_code,
            height=400,
            help="Enter your Pine Script strategy code"
        )
        
        convert_btn = st.button("🔄 Convert to Python", type="primary")
    
    with col2:
        st.subheader("Python Output")
        
        if convert_btn and pinescript_code:
            try:
                with st.spinner("Converting Pine Script..."):
                    python_code, warnings = st.session_state.converter.convert(pinescript_code)
                
                # Store converted code in session state for saving
                st.session_state.converted_code = python_code
                st.session_state.conversion_warnings = warnings
                
                # Show warnings if any
                if warnings:
                    st.warning(f"Conversion completed with {len(warnings)} warnings:")
                    for warning in warnings:
                        st.caption(f"Line {warning.line_number}: {warning.message}")
                
                # Display converted code
                st.code(python_code, language='python')
                
            except ConversionError as e:
                st.error(f"Conversion failed: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
        
        # Show save option if we have converted code
        if 'converted_code' in st.session_state:
            # Option to save strategy
            st.markdown("---")
            col1_save, col2_save = st.columns(2)
            
            with col1_save:
                strategy_name = st.text_input("Strategy Name:", value="ConvertedStrategy")
            
            with col2_save:
                if st.button("💾 Save Strategy"):
                    try:
                        # Check if strategy name already exists
                        existing_strategies = st.session_state.db.get_all_strategies()
                        existing_names = [s.name for s in existing_strategies]
                        
                        if strategy_name in existing_names:
                            st.error(f"❌ Strategy name '{strategy_name}' already exists. Please choose a different name.")
                        elif not strategy_name.strip():
                            st.error("❌ Strategy name cannot be empty.")
                        else:
                            strategy = st.session_state.db.save_strategy(
                                name=strategy_name,
                                code=st.session_state.converted_code,
                                strategy_type="python",
                                description="Converted from Pine Script"
                            )
                            st.success(f"✅ Strategy '{strategy_name}' saved successfully!")
                            # Clear the converted code from session state after successful save
                            del st.session_state.converted_code
                            if 'conversion_warnings' in st.session_state:
                                del st.session_state.conversion_warnings
                    except Exception as e:
                        error_msg = str(e)
                        if "UNIQUE constraint failed" in error_msg:
                            st.error(f"❌ Strategy name '{strategy_name}' already exists. Please choose a different name.")
                        elif "NOT NULL constraint failed" in error_msg:
                            st.error("❌ Strategy name is required.")
                        else:
                            st.error(f"❌ Error saving strategy: {error_msg}")
                        # Log the full error for debugging
                        import logging
                        logging.error(f"Error saving strategy '{strategy_name}': {e}")
            
            # Show a clear button to reset
            if st.button("🔄 Clear Conversion"):
                if 'converted_code' in st.session_state:
                    del st.session_state.converted_code
                if 'conversion_warnings' in st.session_state:
                    del st.session_state.conversion_warnings
                st.rerun()

def show_strategy_manager():
    """Display strategy management interface."""
    st.header("📊 Strategy Manager")
    
    # Get all strategies
    strategies = st.session_state.db.get_all_strategies()
    
    if not strategies:
        st.info("No strategies found. Create one using the Pine Script converter or add manually.")
        return
    
    # Strategy selection
    strategy_names = [s.name for s in strategies]
    selected_strategy_name = st.selectbox("Select Strategy:", strategy_names)
    
    selected_strategy = next(s for s in strategies if s.name == selected_strategy_name)
    
    # Display strategy details
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Strategy Code")
        st.code(selected_strategy.code, language='python')
    
    with col2:
        st.subheader("Strategy Details")
        st.write(f"**Name:** {selected_strategy.name}")
        st.write(f"**Type:** {selected_strategy.strategy_type}")
        st.write(f"**Created:** {selected_strategy.created_at.strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Active:** {'✅' if selected_strategy.is_active else '❌'}")
        
        if selected_strategy.description:
            st.write(f"**Description:** {selected_strategy.description}")
        
        # Parameters
        params = selected_strategy.get_parameters()
        if params:
            st.write("**Parameters:**")
            for key, value in params.items():
                st.write(f"- {key}: {value}")
        
        # Action buttons
        st.markdown("---")
        
        if st.button("🔬 Run Backtest", use_container_width=True):
            st.session_state.selected_strategy_for_backtest = selected_strategy
            st.session_state.current_page = "🔬 Backtesting"
            st.rerun()
        
        if st.button("⚡ Optimize", use_container_width=True):
            st.session_state.selected_strategy_for_optimization = selected_strategy
            st.session_state.current_page = "⚡ Optimization"
            st.rerun()
        
        if st.button("🗑️ Delete Strategy", use_container_width=True):
            if st.session_state.get('confirm_delete', False):
                # Delete strategy
                session = st.session_state.db.get_session()
                try:
                    strategy_to_delete = session.query(Strategy).filter(Strategy.id == selected_strategy.id).first()
                    strategy_to_delete.is_active = False
                    session.commit()
                    st.success("Strategy deleted successfully!")
                    st.rerun()
                finally:
                    session.close()
            else:
                st.session_state.confirm_delete = True
                st.warning("Click again to confirm deletion")

def show_backtesting():
    """Display backtesting interface."""
    st.header("🔬 Strategy Backtesting")
    
    # Strategy selection
    strategies = st.session_state.db.get_all_strategies()
    
    if not strategies:
        st.warning("No strategies available for backtesting.")
        return
    
    # Check if strategy was pre-selected
    if 'selected_strategy_for_backtest' in st.session_state:
        strategy_names = [s.name for s in strategies]
        default_index = strategy_names.index(st.session_state.selected_strategy_for_backtest.name)
        del st.session_state.selected_strategy_for_backtest
    else:
        default_index = 0
    
    selected_strategy_name = st.selectbox(
        "Select Strategy:", 
        [s.name for s in strategies],
        index=default_index
    )
    
    selected_strategy = next(s for s in strategies if s.name == selected_strategy_name)
    
    # Backtest parameters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.selectbox(
            "Symbol:", 
            st.session_state.config.get('data.default_tickers', ['BTCUSDT', 'ETHUSDT']),
            help="Select trading symbol"
        )
    
    with col2:
        timeframe = st.selectbox(
            "Timeframe:",
            ['1d', '1h', '30m', '15m', '5m', '1m'],
            index=0  # Default to 1d (daily data works better)
        )
    
    with col3:
        period = st.selectbox(
            "Period:",
            ['1mo', '3mo', '6mo', '1y', '2y'],
            index=3  # Default to 1y
        )
    
    # Strategy parameters (if any)
    params = selected_strategy.get_parameters()
    if params:
        st.subheader("Strategy Parameters")
        updated_params = {}
        
        param_cols = st.columns(min(len(params), 4))
        for i, (key, value) in enumerate(params.items()):
            with param_cols[i % len(param_cols)]:
                if isinstance(value, int):
                    updated_params[key] = st.number_input(f"{key}:", value=value, step=1)
                elif isinstance(value, float):
                    updated_params[key] = st.number_input(f"{key}:", value=value, step=0.1)
                else:
                    updated_params[key] = st.text_input(f"{key}:", value=str(value))
    else:
        updated_params = {}
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        try:
            with st.spinner("Running backtest..."):
                # Create strategy class from code (simplified)
                # In a real implementation, you'd dynamically import the strategy
                results = st.session_state.backtest_engine.quick_backtest(
                    SimpleMovingAverageStrategy, symbol
                )
                
                if results['success']:
                    st.success("Backtest completed successfully!")
                    
                    # Display results
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Net Profit", f"${results['net_profit']:.2f}")
                    
                    with col2:
                        st.metric("Profit Factor", f"{results['profit_factor']:.2f}")
                    
                    with col3:
                        st.metric("Max Drawdown", f"{results['max_drawdown']:.2%}")
                    
                    with col4:
                        st.metric("Win Rate", f"{results['win_rate']:.2%}")
                    
                    st.metric("Number of Trades", results['number_of_trades'])
                    st.metric("Execution Time", f"{results['execution_time']:.2f}s")
                    
                    # Save backtest results
                    backtest_data = {
                        'strategy_id': selected_strategy.id,
                        'symbol': symbol,
                        'timeframe': timeframe,
                        'start_date': pd.Timestamp.now() - pd.Timedelta(days=365),
                        'end_date': pd.Timestamp.now(),
                        'initial_capital': 10000,
                        'final_capital': 10000 + results['net_profit'],
                        'net_profit': results['net_profit'],
                        'profit_factor': results['profit_factor'],
                        'max_drawdown': results['max_drawdown'],
                        'number_of_trades': results['number_of_trades'],
                        'win_rate': results['win_rate'],
                        'execution_time': results['execution_time']
                    }
                    
                    st.session_state.db.save_backtest(backtest_data)
                    
                else:
                    st.error(f"Backtest failed: {results.get('error', 'Unknown error')}")
                    
        except Exception as e:
            st.error(f"Error running backtest: {e}")

def show_optimization():
    """Display optimization interface."""
    st.header("⚡ Strategy Optimization")
    
    strategies = st.session_state.db.get_all_strategies()
    
    if not strategies:
        st.warning("No strategies available for optimization.")
        return
    
    # Strategy selection
    selected_strategy_name = st.selectbox("Select Strategy:", [s.name for s in strategies])
    selected_strategy = next(s for s in strategies if s.name == selected_strategy_name)
    
    # Optimization parameters
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "Symbol:", 
            st.session_state.config.get('data.default_tickers', ['BTCUSDT'])
        )
        
        algorithm = st.selectbox(
            "Algorithm:",
            ['random_search', 'bayesian']
        )
        
        objective = st.selectbox(
            "Objective:",
            ['profit_factor', 'sharpe_ratio', 'net_profit', 'win_rate']
        )
    
    with col2:
        n_iterations = st.number_input("Iterations:", min_value=10, max_value=200, value=50)
        walk_forward = st.checkbox("Use Walk-Forward Validation", value=True)
    
    # Parameter space definition
    st.subheader("Parameter Space")
    st.write("Define the parameter ranges for optimization:")
    
    # For demonstration, we'll use a simple parameter space
    param_space = ParameterSpace()
    param_space.add_integer('fast_period', 5, 20)
    param_space.add_integer('slow_period', 20, 50)
    
    st.info("Parameter space: fast_period [5-20], slow_period [20-50]")
    
    # Run optimization
    if st.button("🚀 Start Optimization", type="primary"):
        try:
            with st.spinner(f"Running {algorithm} optimization with {n_iterations} iterations..."):
                # For demo, we'll use a simple strategy
                results = st.session_state.optimizer.optimize_strategy(
                    SimpleMovingAverageStrategy,
                    symbol,
                    param_space,
                    algorithm=algorithm,
                    objective=objective,
                    walk_forward=walk_forward
                )
                
                # Check if results is None or invalid
                if results is None:
                    st.error("Optimization failed: No results returned")
                elif not hasattr(results, 'best_params') or results.best_params is None:
                    st.error("Optimization failed: Invalid results returned")
                else:
                    st.success("Optimization completed!")
                    
                    # Display results
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("Best Parameters")
                        if results.best_params:
                            for param, value in results.best_params.items():
                                st.write(f"**{param}:** {value}")
                        else:
                            st.write("No best parameters found")
                    
                    with col2:
                        st.subheader("Best Score")
                        st.metric(objective.replace('_', ' ').title(), f"{results.best_score:.4f}")
                        st.metric("Execution Time", f"{results.execution_time:.2f}s")
                        st.metric("Iterations", results.iterations)
                    
                    # Show optimization progress
                    if results.all_results and len(results.all_results) > 0:
                        st.subheader("Optimization Progress")
                        
                        df_results = pd.DataFrame([
                            {'iteration': r['iteration'], 'score': r['score']} 
                            for r in results.all_results
                        ])
                        
                        fig = px.line(df_results, x='iteration', y='score', 
                                    title=f'{objective.title()} Over Iterations')
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("No optimization progress data available")
                
        except Exception as e:
            st.error(f"Optimization failed: {e}")
            import traceback
            st.error(f"Details: {traceback.format_exc()}")

def show_live_signals():
    """Display live signals interface."""
    st.header("📡 Live Trading Signals")
    
    # Configuration status
    signals_enabled = st.session_state.config.is_signals_enabled()
    telegram_enabled = st.session_state.config.is_telegram_enabled()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Signal Generation", "✅ Active" if signals_enabled else "❌ Inactive")
    
    with col2:
        st.metric("Telegram Bot", "✅ Connected" if telegram_enabled else "❌ Disconnected")
    
    if not signals_enabled:
        st.warning("Live signals are disabled. Enable them in Settings to start generating signals.")
        return
    
    # Open signals
    open_signals = st.session_state.db.get_open_signals()
    
    if open_signals:
        st.subheader("Open Signals")
        
        signal_data = []
        for signal in open_signals:
            signal_data.append({
                'Strategy': signal.strategy.name,
                'Symbol': signal.symbol,
                'Type': signal.signal_type,
                'Price': f"${signal.price:.4f}",
                'Take Profit': f"${signal.take_profit:.4f}" if signal.take_profit else "N/A",
                'Stop Loss': f"${signal.stop_loss:.4f}" if signal.stop_loss else "N/A",
                'Created': signal.created_at.strftime('%Y-%m-%d %H:%M')
            })
        
        st.dataframe(pd.DataFrame(signal_data), use_container_width=True)
    else:
        st.info("No open signals currently.")
    
    # Signal history and performance
    st.subheader("Signal Performance")
    
    strategies = st.session_state.db.get_all_strategies()
    if strategies:
        strategy_perf = []
        for strategy in strategies:
            perf = st.session_state.db.get_strategy_performance(strategy.id)
            if perf['total_signals'] > 0:
                strategy_perf.append({
                    'Strategy': strategy.name,
                    'Total Signals': perf['total_signals'],
                    'Win Rate': f"{perf['win_rate']:.2%}",
                    'Total PnL': f"${perf['total_pnl']:.2f}",
                    'Avg PnL': f"${perf['avg_pnl']:.2f}"
                })
        
        if strategy_perf:
            st.dataframe(pd.DataFrame(strategy_perf), use_container_width=True)
        else:
            st.info("No signal history available.")

def show_settings():
    """Display settings interface."""
    st.header("⚙️ Application Settings")
    
    # Configuration sections
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Sources", "🔬 Backtesting", "📡 Telegram", "⚡ Optimization"])
    
    with tab1:
        st.subheader("Data Source Configuration")
        
        # Yahoo Finance settings
        yf_enabled = st.checkbox("Enable Yahoo Finance", 
                                value=st.session_state.config.get('data.yahoo_finance.enabled', True))
        
        # Binance settings
        binance_enabled = st.checkbox("Enable Binance", 
                                     value=st.session_state.config.get('data.binance.enabled', False))
        
        if binance_enabled:
            api_key = st.text_input("Binance API Key:", 
                                   value=st.session_state.config.get('data.binance.api_key', ''),
                                   type="password")
            api_secret = st.text_input("Binance API Secret:", 
                                      value=st.session_state.config.get('data.binance.api_secret', ''),
                                      type="password")
            testnet = st.checkbox("Use Testnet", 
                                 value=st.session_state.config.get('data.binance.testnet', True))
        
        # Default tickers
        default_tickers = st.text_area("Default Tickers (one per line):",
                                      value='\n'.join(st.session_state.config.get('data.default_tickers', [])))
    
    with tab2:
        st.subheader("Backtesting Configuration")
        
        initial_capital = st.number_input("Initial Capital ($):",
                                         value=st.session_state.config.get('backtesting.initial_capital', 10000))
        
        commission = st.number_input("Commission (%):",
                                    value=st.session_state.config.get('backtesting.commission', 0.1),
                                    step=0.01)
        
        slippage = st.number_input("Slippage (%):",
                                  value=st.session_state.config.get('backtesting.slippage', 0.05),
                                  step=0.01)
    
    with tab3:
        st.subheader("Telegram Bot Configuration")
        
        telegram_enabled = st.checkbox("Enable Telegram Bot",
                                      value=st.session_state.config.get('telegram.enabled', False))
        
        if telegram_enabled:
            bot_token = st.text_input("Bot Token:",
                                     value=st.session_state.config.get('telegram.bot_token', ''),
                                     type="password")
            
            chat_id = st.text_input("Chat ID:",
                                   value=st.session_state.config.get('telegram.chat_id', ''))
        
        # Signal settings
        signals_enabled = st.checkbox("Enable Live Signals",
                                     value=st.session_state.config.get('signals.enabled', False))
        
        if signals_enabled:
            update_interval = st.number_input("Update Interval (seconds):",
                                             value=st.session_state.config.get('signals.update_interval', 300))
    
    with tab4:
        st.subheader("Optimization Configuration")
        
        algorithm = st.selectbox("Default Algorithm:",
                                ['random_search', 'bayesian'],
                                index=0 if st.session_state.config.get('optimization.algorithm') == 'random_search' else 1)
        
        max_iterations = st.number_input("Max Iterations:",
                                        value=st.session_state.config.get('optimization.max_iterations', 100))
        
        # Walk-forward settings
        st.subheader("Walk-Forward Validation")
        train_ratio = st.slider("Training Ratio:",
                               value=st.session_state.config.get('optimization.walk_forward.train_ratio', 0.7),
                               min_value=0.5, max_value=0.9, step=0.05)
        
        validation_ratio = st.slider("Validation Ratio:",
                                    value=st.session_state.config.get('optimization.walk_forward.validation_ratio', 0.15),
                                    min_value=0.1, max_value=0.3, step=0.05)
    
    # Save settings button
    if st.button("💾 Save Settings", type="primary"):
        try:
            # Update configuration
            st.session_state.config.set('data.yahoo_finance.enabled', yf_enabled)
            st.session_state.config.set('data.binance.enabled', binance_enabled)
            
            if binance_enabled:
                st.session_state.config.set('data.binance.api_key', api_key)
                st.session_state.config.set('data.binance.api_secret', api_secret)
                st.session_state.config.set('data.binance.testnet', testnet)
            
            # Update tickers
            tickers = [t.strip() for t in default_tickers.split('\n') if t.strip()]
            st.session_state.config.set('data.default_tickers', tickers)
            
            # Backtesting settings
            st.session_state.config.set('backtesting.initial_capital', initial_capital)
            st.session_state.config.set('backtesting.commission', commission / 100)
            st.session_state.config.set('backtesting.slippage', slippage / 100)
            
            # Telegram settings
            st.session_state.config.set('telegram.enabled', telegram_enabled)
            if telegram_enabled:
                st.session_state.config.set('telegram.bot_token', bot_token)
                st.session_state.config.set('telegram.chat_id', chat_id)
            
            # Signal settings
            st.session_state.config.set('signals.enabled', signals_enabled)
            if signals_enabled:
                st.session_state.config.set('signals.update_interval', update_interval)
            
            # Optimization settings
            st.session_state.config.set('optimization.algorithm', algorithm)
            st.session_state.config.set('optimization.max_iterations', max_iterations)
            st.session_state.config.set('optimization.walk_forward.train_ratio', train_ratio)
            st.session_state.config.set('optimization.walk_forward.validation_ratio', validation_ratio)
            
            # Save to file
            st.session_state.config.save_config()
            
            st.success("Settings saved successfully!")
            
        except Exception as e:
            st.error(f"Error saving settings: {e}")

if __name__ == "__main__":
    main()