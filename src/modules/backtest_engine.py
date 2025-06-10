"""
Backtesting Engine for TradingView Strategy Generator
Handles strategy backtesting using backtrader with performance metrics calculation.
"""

import backtrader as bt
import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class BacktestResults:
    """Container for backtest results."""
    strategy_name: str
    symbol: str
    timeframe: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    net_profit: float
    profit_factor: float
    max_drawdown: float
    sharpe_ratio: float
    number_of_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    risk_reward_ratio: float
    execution_time: float
    trades: List[Dict[str, Any]]

class PerformanceAnalyzer(bt.Analyzer):
    """Custom analyzer for comprehensive performance metrics."""
    
    def __init__(self):
        self.trades_analysis = []
        self.returns = []
        self.equity_curve = []
        
    def next(self):
        """Track equity curve."""
        self.equity_curve.append(self.strategy.broker.getvalue())
    
    def notify_trade(self, trade):
        """Track individual trades."""
        if trade.isclosed:
            self.trades_analysis.append({
                'entry_date': bt.num2date(trade.dtopen),
                'exit_date': bt.num2date(trade.dtclose),
                'entry_price': trade.price,
                'exit_price': trade.pnl / trade.size + trade.price,
                'size': trade.size,
                'pnl': trade.pnl,
                'pnl_net': trade.pnlcomm,
                'commission': trade.commission,
                'duration': trade.barlen
            })
    
    def get_analysis(self):
        """Return comprehensive analysis."""
        if not self.trades_analysis:
            return self._empty_analysis()
        
        # Calculate metrics
        total_trades = len(self.trades_analysis)
        winning_trades = [t for t in self.trades_analysis if t['pnl'] > 0]
        losing_trades = [t for t in self.trades_analysis if t['pnl'] < 0]
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        total_gross_profit = sum(t['pnl'] for t in winning_trades)
        total_gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        
        profit_factor = total_gross_profit / total_gross_loss if total_gross_loss > 0 else float('inf')
        
        # Calculate drawdown
        equity_curve = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(np.min(drawdown))
        
        # Calculate Sharpe ratio
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'trades': self.trades_analysis,
            'equity_curve': self.equity_curve
        }
    
    def _empty_analysis(self):
        """Return empty analysis when no trades."""
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0,
            'trades': [],
            'equity_curve': []
        }

class DataFeed:
    """Handles data fetching from various sources."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_yahoo_data(self, symbol: str, period: str = "1y", interval: str = "5m") -> pd.DataFrame:
        """Fetch data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                raise ValueError(f"No data found for {symbol}")
            
            # Ensure proper column names for backtrader
            data.columns = [col.lower() for col in data.columns]
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching Yahoo Finance data: {e}")
            raise
    
    def prepare_backtrader_data(self, df: pd.DataFrame) -> bt.feeds.PandasData:
        """Convert pandas DataFrame to backtrader data feed."""
        return bt.feeds.PandasData(
            dataname=df,
            datetime=None,  # Use index as datetime
            open='open',
            high='high',
            low='low',
            close='close',
            volume='volume',
            openinterest=None
        )

class BacktestEngine:
    """Main backtesting engine using backtrader."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_feed = DataFeed()
    
    def run_backtest(self, strategy_class, symbol: str, parameters: Dict[str, Any] = None,
                    start_date: str = None, end_date: str = None, 
                    timeframe: str = "5m") -> BacktestResults:
        """
        Run backtest for a strategy.
        
        Args:
            strategy_class: The strategy class to test
            symbol: Trading symbol (e.g., 'BTCUSDT')
            parameters: Strategy parameters
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe
            
        Returns:
            BacktestResults object
        """
        start_time = time.time()
        
        try:
            # Initialize cerebro
            cerebro = bt.Cerebro()
            
            # Set initial capital
            initial_capital = self.config.get('initial_capital', 10000)
            cerebro.broker.setcash(initial_capital)
            
            # Set commission
            commission = self.config.get('commission', 0.001)
            cerebro.broker.setcommission(commission=commission)
            
            # Add strategy with parameters
            if parameters:
                # Create strategy class with custom parameters
                strategy_params = parameters
                cerebro.addstrategy(strategy_class, **strategy_params)
            else:
                cerebro.addstrategy(strategy_class)
            
            # Fetch and add data
            data = self._get_data(symbol, start_date, end_date, timeframe)
            cerebro.adddata(data)
            
            # Add analyzers
            cerebro.addanalyzer(PerformanceAnalyzer, _name='performance')
            cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
            cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
            cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe')
            
            # Run backtest
            results = cerebro.run()
            final_capital = cerebro.broker.getvalue()
            
            # Extract results
            strategy_result = results[0]
            performance = strategy_result.analyzers.performance.get_analysis()
            
            execution_time = time.time() - start_time
            
            # Create results object
            backtest_results = BacktestResults(
                strategy_name=strategy_class.__name__,
                symbol=symbol,
                timeframe=timeframe,
                start_date=datetime.strptime(start_date, '%Y-%m-%d') if start_date else datetime.now() - timedelta(days=365),
                end_date=datetime.strptime(end_date, '%Y-%m-%d') if end_date else datetime.now(),
                initial_capital=initial_capital,
                final_capital=final_capital,
                net_profit=final_capital - initial_capital,
                profit_factor=performance.get('profit_factor', 0),
                max_drawdown=performance.get('max_drawdown', 0),
                sharpe_ratio=performance.get('sharpe_ratio', 0),
                number_of_trades=performance.get('total_trades', 0),
                win_rate=performance.get('win_rate', 0),
                avg_win=performance.get('avg_win', 0),
                avg_loss=performance.get('avg_loss', 0),
                risk_reward_ratio=abs(performance.get('avg_win', 0) / performance.get('avg_loss', 1)) if performance.get('avg_loss', 0) != 0 else 0,
                execution_time=execution_time,
                trades=performance.get('trades', [])
            )
            
            self.logger.info(f"Backtest completed for {symbol} in {execution_time:.2f} seconds")
            return backtest_results
            
        except Exception as e:
            self.logger.error(f"Backtest failed: {e}")
            raise BacktestError(f"Backtest failed: {e}")
    
    def _get_data(self, symbol: str, start_date: str = None, end_date: str = None, 
                  timeframe: str = "5m") -> bt.feeds.PandasData:
        """Get data for backtesting."""
        
        # Convert timeframe to Yahoo Finance format
        yf_interval = self._convert_timeframe(timeframe)
        
        # Calculate period if dates not provided
        if not start_date or not end_date:
            period = "1y"  # Default to 1 year
            df = self.data_feed.get_yahoo_data(symbol, period=period, interval=yf_interval)
        else:
            # For date ranges, we need to use period approach as yfinance doesn't support date ranges well with intervals
            df = self.data_feed.get_yahoo_data(symbol, period="2y", interval=yf_interval)
            
            # Filter by date range if provided
            if start_date:
                df = df[df.index >= start_date]
            if end_date:
                df = df[df.index <= end_date]
        
        return self.data_feed.prepare_backtrader_data(df)
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert timeframe to Yahoo Finance format."""
        timeframe_map = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '1d': '1d'
        }
        return timeframe_map.get(timeframe, '5m')
    
    def validate_strategy(self, strategy_class) -> bool:
        """Validate that strategy class is properly formed."""
        try:
            # Check if it's a valid backtrader strategy
            if not issubclass(strategy_class, bt.Strategy):
                return False
            
            # Check for required methods
            required_methods = ['__init__', 'next']
            for method in required_methods:
                if not hasattr(strategy_class, method):
                    return False
            
            return True
            
        except Exception:
            return False
    
    def quick_backtest(self, strategy_class, symbol: str = "BTCUSDT") -> Dict[str, Any]:
        """Run a quick backtest with default parameters."""
        try:
            results = self.run_backtest(strategy_class, symbol)
            
            return {
                'success': True,
                'net_profit': results.net_profit,
                'profit_factor': results.profit_factor,
                'max_drawdown': results.max_drawdown,
                'number_of_trades': results.number_of_trades,
                'win_rate': results.win_rate,
                'execution_time': results.execution_time
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

class BacktestError(Exception):
    """Exception raised when backtesting fails."""
    pass

# Example strategy for testing
class SimpleMovingAverageStrategy(bt.Strategy):
    """Simple Moving Average crossover strategy for testing."""
    
    params = (
        ('fast_period', 10),
        ('slow_period', 30),
    )
    
    def __init__(self):
        self.fast_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SimpleMovingAverage(
            self.data.close, period=self.params.slow_period
        )
        self.crossover = bt.indicators.CrossOver(self.fast_ma, self.slow_ma)
    
    def next(self):
        if not self.position:
            if self.crossover > 0:  # Fast MA crosses above slow MA
                self.buy()
        else:
            if self.crossover < 0:  # Fast MA crosses below slow MA
                self.close()