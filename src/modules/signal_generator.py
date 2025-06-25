"""
Signal Generator Module
Generates live trading signals from strategies using real-time market data.
"""

import logging
import time
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Type
import pandas as pd
import yfinance as yf
from sqlalchemy.orm import sessionmaker

from core.database import DatabaseManager, Strategy, Signal
from modules.backtest_engine import BacktestEngine, DataFeed


class SignalGenerator:
    """Generates live trading signals from strategies."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        self.data_feed = DataFeed()
        self._running = False
        self._signal_thread = None
        
    def start_signal_generation(self, symbols: List[str], update_interval: int = 300):
        """Start continuous signal generation in background."""
        if self._running:
            self.logger.warning("Signal generation already running")
            return
            
        self._running = True
        self._signal_thread = threading.Thread(
            target=self._signal_generation_loop,
            args=(symbols, update_interval),
            daemon=True
        )
        self._signal_thread.start()
        self.logger.info(f"Started signal generation for symbols: {symbols}")
    
    def stop_signal_generation(self):
        """Stop signal generation."""
        self._running = False
        if self._signal_thread:
            self._signal_thread.join(timeout=10)
        self.logger.info("Stopped signal generation")
    
    def _signal_generation_loop(self, symbols: List[str], update_interval: int):
        """Main signal generation loop."""
        while self._running:
            try:
                self.generate_all_signals(symbols)
                time.sleep(update_interval)
            except Exception as e:
                self.logger.error(f"Error in signal generation loop: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def generate_all_signals(self, symbols: List[str]) -> List[Signal]:
        """Generate signals for all active strategies."""
        all_signals = []
        
        # Get all active strategies
        strategies = self.db.get_all_strategies()
        active_strategies = [s for s in strategies if s.is_active]
        
        for strategy in active_strategies:
            try:
                signals = self.generate_signals_for_strategy(strategy, symbols)
                all_signals.extend(signals)
            except Exception as e:
                self.logger.error(f"Error generating signals for strategy {strategy.name}: {e}")
        
        return all_signals
    
    def generate_signals_for_strategy(self, strategy: Strategy, symbols: List[str], timeframe: str = '1d') -> List[Signal]:
        """Generate signals for a specific strategy."""
        signals = []
        
        for symbol in symbols:
            try:
                # Get recent market data
                data = self._get_market_data(symbol, timeframe)
                if data is None or data.empty:
                    continue
                
                # Analyze strategy conditions
                signal_info = self._analyze_strategy_conditions(strategy, data, symbol)
                
                if signal_info:
                    # Create and save signal
                    signal = self._create_signal(strategy, symbol, signal_info)
                    if signal:
                        signals.append(signal)
                        self.logger.info(f"Generated {signal_info['type']} signal for {symbol} using {strategy.name}")
                        
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol} with strategy {strategy.name}: {e}")
        
        return signals
    
    def _get_market_data(self, symbol: str, timeframe: str = '1d', periods: int = 100) -> Optional[pd.DataFrame]:
        """Get recent market data for analysis."""
        try:
            # Convert symbol format if needed (e.g., BTCUSDT -> BTC-USD for Yahoo)
            yahoo_symbol = self._convert_symbol_format(symbol)
            
            # Get recent data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=periods)
            
            data = self.data_feed.get_yahoo_data(
                symbol=yahoo_symbol,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return None
    
    def _convert_symbol_format(self, symbol: str) -> str:
        """Convert symbol format for different exchanges."""
        # Convert common crypto symbols to Yahoo Finance format
        symbol_map = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'ADAUSDT': 'ADA-USD',
            'BNBUSDT': 'BNB-USD',
            'XRPUSDT': 'XRP-USD',
            'SOLUSDT': 'SOL-USD',
            'DOTUSDT': 'DOT-USD',
            'LINKUSDT': 'LINK-USD',
        }
        
        return symbol_map.get(symbol, symbol)
    
    def _analyze_strategy_conditions(self, strategy: Strategy, data: pd.DataFrame, symbol: str) -> Optional[Dict[str, Any]]:
        """Analyze if strategy conditions are met for signal generation."""
        try:
            # For now, implement a simple moving average crossover detection
            # In a full implementation, this would dynamically execute the strategy code
            
            # Calculate moving averages
            data['sma_10'] = data['close'].rolling(window=10).mean()
            data['sma_30'] = data['close'].rolling(window=30).mean()
            
            # Get last few rows for analysis
            recent_data = data.tail(3)
            if len(recent_data) < 3:
                return None
            
            # Check for crossover patterns
            current_price = recent_data['close'].iloc[-1]
            prev_fast = recent_data['sma_10'].iloc[-2]
            prev_slow = recent_data['sma_30'].iloc[-2]
            curr_fast = recent_data['sma_10'].iloc[-1]
            curr_slow = recent_data['sma_30'].iloc[-1]
            
            # Check if we already have an open signal for this strategy and symbol
            existing_signals = self.db.get_open_signals(strategy.id)
            has_open_signal = any(s.symbol == symbol for s in existing_signals)
            
            if has_open_signal:
                return None  # Don't generate new signal if one is already open
            
            # Bullish crossover (fast MA crosses above slow MA)
            if prev_fast <= prev_slow and curr_fast > curr_slow:
                return {
                    'type': 'BUY',
                    'price': current_price,
                    'take_profit': current_price * 1.05,  # 5% profit target
                    'stop_loss': current_price * 0.97,    # 3% stop loss
                    'quantity': 1.0,
                    'confidence': self._calculate_signal_confidence(recent_data)
                }
            
            # Bearish crossover (fast MA crosses below slow MA)
            elif prev_fast >= prev_slow and curr_fast < curr_slow:
                return {
                    'type': 'SELL',
                    'price': current_price,
                    'take_profit': current_price * 0.95,  # 5% profit target for short
                    'stop_loss': current_price * 1.03,    # 3% stop loss for short
                    'quantity': 1.0,
                    'confidence': self._calculate_signal_confidence(recent_data)
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error analyzing strategy conditions: {e}")
            return None
    
    def _calculate_signal_confidence(self, data: pd.DataFrame) -> float:
        """Calculate confidence score for the signal."""
        try:
            # Simple confidence calculation based on volume and volatility
            recent_volume = data['volume'].tail(5).mean()
            avg_volume = data['volume'].mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1.0
            
            # Higher volume = higher confidence (capped at 2x)
            volume_confidence = min(volume_ratio, 2.0) / 2.0
            
            # Lower volatility = higher confidence
            returns = data['close'].pct_change().tail(10)
            volatility = returns.std()
            volatility_confidence = max(0.1, 1.0 - min(volatility * 10, 0.9))
            
            # Combine factors
            confidence = (volume_confidence * 0.4 + volatility_confidence * 0.6)
            return round(confidence, 2)
            
        except Exception:
            return 0.5  # Default confidence
    
    def _create_signal(self, strategy: Strategy, symbol: str, signal_info: Dict[str, Any]) -> Optional[Signal]:
        """Create and save a new signal."""
        try:
            signal_data = {
                'strategy_id': strategy.id,
                'symbol': symbol,
                'signal_type': signal_info['type'],
                'price': signal_info['price'],
                'take_profit': signal_info.get('take_profit'),
                'stop_loss': signal_info.get('stop_loss'),
                'quantity': signal_info.get('quantity', 1.0),
            }
            
            signal = self.db.save_signal(signal_data)
            return signal
            
        except Exception as e:
            self.logger.error(f"Error creating signal: {e}")
            return None
    
    def check_signal_exits(self, symbols: List[str]):
        """Check if any open signals should be closed."""
        open_signals = self.db.get_open_signals()
        
        for signal in open_signals:
            try:
                # Get current market data
                data = self._get_market_data(signal.symbol, periods=5)
                if data is None or data.empty:
                    continue
                
                current_price = data['close'].iloc[-1]
                
                # Check exit conditions
                should_close, close_reason = self._check_exit_conditions(signal, current_price)
                
                if should_close:
                    signal.close_signal(current_price)
                    
                    # Update in database
                    session = self.db.get_session()
                    try:
                        session.merge(signal)
                        session.commit()
                        self.logger.info(f"Closed signal {signal.id} for {signal.symbol}: {close_reason}")
                    finally:
                        session.close()
                        
            except Exception as e:
                self.logger.error(f"Error checking exit conditions for signal {signal.id}: {e}")
    
    def _check_exit_conditions(self, signal: Signal, current_price: float) -> tuple[bool, str]:
        """Check if signal should be closed based on current price."""
        # Take profit hit
        if signal.take_profit:
            if signal.signal_type == 'BUY' and current_price >= signal.take_profit:
                return True, "Take profit reached"
            elif signal.signal_type == 'SELL' and current_price <= signal.take_profit:
                return True, "Take profit reached"
        
        # Stop loss hit
        if signal.stop_loss:
            if signal.signal_type == 'BUY' and current_price <= signal.stop_loss:
                return True, "Stop loss triggered"
            elif signal.signal_type == 'SELL' and current_price >= signal.stop_loss:
                return True, "Stop loss triggered"
        
        # Time-based exit (optional - close signals older than 24 hours)
        if signal.created_at:
            age_hours = (datetime.now() - signal.created_at).total_seconds() / 3600
            if age_hours > 24:
                return True, "Time-based exit (24h limit)"
        
        return False, ""