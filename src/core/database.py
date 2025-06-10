"""
Database models for TradingView Strategy Generator
Defines the database schema for strategies, signals, and performance tracking.
"""

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
import json
from typing import Dict, Any, List, Optional

Base = declarative_base()

class Strategy(Base):
    """Model for storing trading strategies."""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text)
    strategy_type = Column(String(50))  # 'pinescript', 'python'
    code = Column(Text, nullable=False)
    parameters = Column(Text)  # JSON string of strategy parameters
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    # Relationships
    backtests = relationship("Backtest", back_populates="strategy")
    signals = relationship("Signal", back_populates="strategy")
    
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set strategy parameters as JSON string."""
        self.parameters = json.dumps(params)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters from JSON string."""
        if self.parameters:
            return json.loads(self.parameters)
        return {}

class Backtest(Base):
    """Model for storing backtest results."""
    __tablename__ = 'backtests'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    
    # Performance Metrics
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    net_profit = Column(Float, nullable=False)
    profit_factor = Column(Float)
    max_drawdown = Column(Float)
    sharpe_ratio = Column(Float)
    number_of_trades = Column(Integer, default=0)
    win_rate = Column(Float)
    avg_win = Column(Float)
    avg_loss = Column(Float)
    risk_reward_ratio = Column(Float)
    
    # Metadata
    execution_time = Column(Float)  # Time taken to run backtest in seconds
    created_at = Column(DateTime, default=datetime.utcnow)
    trades_data = Column(Text)  # JSON string of individual trades
    
    # Relationships
    strategy = relationship("Strategy", back_populates="backtests")
    
    def set_trades_data(self, trades: List[Dict[str, Any]]) -> None:
        """Set trades data as JSON string."""
        self.trades_data = json.dumps(trades)
    
    def get_trades_data(self) -> List[Dict[str, Any]]:
        """Get trades data from JSON string."""
        if self.trades_data:
            return json.loads(self.trades_data)
        return []

class Signal(Base):
    """Model for storing live trading signals."""
    __tablename__ = 'signals'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    symbol = Column(String(20), nullable=False)
    signal_type = Column(String(10), nullable=False)  # 'BUY', 'SELL', 'CLOSE'
    price = Column(Float, nullable=False)
    take_profit = Column(Float)
    stop_loss = Column(Float)
    quantity = Column(Float)
    
    # Status tracking
    status = Column(String(20), default='OPEN')  # 'OPEN', 'CLOSED', 'CANCELLED'
    created_at = Column(DateTime, default=datetime.utcnow)
    closed_at = Column(DateTime)
    close_price = Column(Float)
    pnl = Column(Float)
    is_profitable = Column(Boolean)
    
    # Telegram integration
    telegram_message_id = Column(String(50))
    is_sent_to_telegram = Column(Boolean, default=False)
    
    # Relationships
    strategy = relationship("Strategy", back_populates="signals")
    
    def close_signal(self, close_price: float, close_time: datetime = None) -> None:
        """Close the signal and calculate P&L."""
        self.close_price = close_price
        self.closed_at = close_time or datetime.now()
        self.status = 'CLOSED'
        
        # Calculate P&L
        if self.signal_type.upper() == 'BUY':
            self.pnl = (close_price - self.price) * (self.quantity or 1)
        else:  # SELL
            self.pnl = (self.price - close_price) * (self.quantity or 1)
        
        self.is_profitable = self.pnl > 0

class Optimization(Base):
    """Model for storing optimization results."""
    __tablename__ = 'optimizations'
    
    id = Column(Integer, primary_key=True)
    strategy_id = Column(Integer, ForeignKey('strategies.id'), nullable=False)
    algorithm = Column(String(50), nullable=False)  # 'random_search', 'bayesian'
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    
    # Optimization parameters
    param_space = Column(Text)  # JSON string of parameter space
    best_params = Column(Text)  # JSON string of best parameters found
    best_score = Column(Float)
    iterations = Column(Integer)
    
    # Timing
    execution_time = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Results
    results_data = Column(Text)  # JSON string of all optimization results
    
    def set_param_space(self, param_space: Dict[str, Any]) -> None:
        """Set parameter space as JSON string."""
        self.param_space = json.dumps(param_space)
    
    def get_param_space(self) -> Dict[str, Any]:
        """Get parameter space from JSON string."""
        if self.param_space:
            return json.loads(self.param_space)
        return {}
    
    def set_best_params(self, params: Dict[str, Any]) -> None:
        """Set best parameters as JSON string."""
        self.best_params = json.dumps(params)
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best parameters from JSON string."""
        if self.best_params:
            return json.loads(self.best_params)
        return {}

class DatabaseManager:
    """Manages database connections and operations."""
    
    def __init__(self, database_url: str = "sqlite:///data/strategies.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
    
    def create_tables(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get database session."""
        return self.SessionLocal()
    
    def save_strategy(self, name: str, code: str, strategy_type: str = "python", 
                     description: str = "", parameters: Dict[str, Any] = None) -> Strategy:
        """Save a new strategy to database."""
        session = self.get_session()
        try:
            strategy = Strategy(
                name=name,
                code=code,
                strategy_type=strategy_type,
                description=description
            )
            if parameters:
                strategy.set_parameters(parameters)
            
            session.add(strategy)
            session.commit()
            session.refresh(strategy)
            return strategy
        finally:
            session.close()
    
    def get_strategy(self, strategy_id: int) -> Optional[Strategy]:
        """Get strategy by ID."""
        session = self.get_session()
        try:
            return session.query(Strategy).filter(Strategy.id == strategy_id).first()
        finally:
            session.close()
    
    def get_all_strategies(self) -> List[Strategy]:
        """Get all active strategies."""
        session = self.get_session()
        try:
            return session.query(Strategy).filter(Strategy.is_active == True).all()
        finally:
            session.close()
    
    def save_backtest(self, backtest_data: Dict[str, Any]) -> Backtest:
        """Save backtest results to database."""
        session = self.get_session()
        try:
            backtest = Backtest(**backtest_data)
            session.add(backtest)
            session.commit()
            session.refresh(backtest)
            return backtest
        finally:
            session.close()
    
    def save_signal(self, signal_data: Dict[str, Any]) -> Signal:
        """Save trading signal to database."""
        session = self.get_session()
        try:
            signal = Signal(**signal_data)
            session.add(signal)
            session.commit()
            session.refresh(signal)
            return signal
        finally:
            session.close()
    
    def get_open_signals(self, strategy_id: Optional[int] = None) -> List[Signal]:
        """Get all open signals, optionally filtered by strategy."""
        session = self.get_session()
        try:
            query = session.query(Signal).filter(Signal.status == 'OPEN')
            if strategy_id:
                query = query.filter(Signal.strategy_id == strategy_id)
            return query.all()
        finally:
            session.close()
    
    def get_strategy_performance(self, strategy_id: int, symbol: str = None) -> Dict[str, Any]:
        """Get performance statistics for a strategy."""
        session = self.get_session()
        try:
            query = session.query(Signal).filter(
                Signal.strategy_id == strategy_id,
                Signal.status == 'CLOSED'
            )
            if symbol:
                query = query.filter(Signal.symbol == symbol)
            
            signals = query.all()
            
            if not signals:
                return {"total_signals": 0, "win_rate": 0, "total_pnl": 0}
            
            total_signals = len(signals)
            profitable_signals = sum(1 for s in signals if s.is_profitable)
            win_rate = profitable_signals / total_signals
            total_pnl = sum(s.pnl or 0 for s in signals)
            
            return {
                "total_signals": total_signals,
                "profitable_signals": profitable_signals,
                "win_rate": win_rate,
                "total_pnl": total_pnl,
                "avg_pnl": total_pnl / total_signals if total_signals > 0 else 0
            }
        finally:
            session.close()