"""
Configuration Manager for TradingView Strategy Generator
Handles loading and managing application configuration from YAML files.
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional
from pathlib import Path

class ConfigManager:
    """Manages application configuration from YAML files."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        self._load_config()
        self._setup_logging()
    
    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as file:
                    self.config = yaml.safe_load(file)
                print(f"Configuration loaded from {self.config_path}")
            else:
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
        except Exception as e:
            print(f"Error loading configuration: {e}")
            self.config = self._get_default_config()
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = self.get('app.log_level', 'INFO')
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('logs/app.log'),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value using dot notation."""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save_config(self) -> None:
        """Save current configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w') as file:
                yaml.dump(self.config, file, default_flow_style=False, indent=2)
            print(f"Configuration saved to {self.config_path}")
        except Exception as e:
            print(f"Error saving configuration: {e}")
    
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._load_config()
        print(f"Configuration reloaded from {self.config_path}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default configuration if file is not found."""
        return {
            'app': {
                'name': 'TradingView Strategy Generator',
                'version': '1.0.0',
                'log_level': 'INFO'
            },
            'data': {
                'yahoo_finance': {'enabled': True},
                'binance': {'enabled': False}
            },
            'backtesting': {
                'initial_capital': 10000,
                'commission': 0.001
            }
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration."""
        return self.get('database', {})
    
    def get_telegram_config(self) -> Dict[str, Any]:
        """Get Telegram bot configuration."""
        return self.get('telegram', {})
    
    def get_backtesting_config(self) -> Dict[str, Any]:
        """Get backtesting configuration."""
        return self.get('backtesting', {})
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """Get optimization configuration."""
        return self.get('optimization', {})
    
    def is_telegram_enabled(self) -> bool:
        """Check if Telegram bot is enabled."""
        return self.get('telegram.enabled', False)
    
    def is_signals_enabled(self) -> bool:
        """Check if live signals are enabled."""
        return self.get('signals.enabled', False)