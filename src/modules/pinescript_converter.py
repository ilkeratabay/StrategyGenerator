"""
Pine Script to Python Converter
Converts Pine Script strategies to Python using backtrader and TA libraries.
"""

import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ConversionWarning:
    """Represents a warning during conversion."""
    line_number: int
    message: str
    original_code: str
    suggested_fix: str = ""

class PineScriptConverter:
    """Converts Pine Script code to Python backtrader strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.warnings: List[ConversionWarning] = []
        
        # Pine Script to Python function mappings
        self.function_mappings = {
            'sma': 'ta.SMA',
            'ema': 'ta.EMA',
            'rsi': 'ta.RSI',
            'macd': 'self._calculate_macd',
            'bb': 'self._calculate_bollinger_bands',
            'stoch': 'self._calculate_stochastic',
            'atr': 'ta.ATR',
            'adx': 'ta.ADX',
            'cci': 'ta.CCI',
            'willr': 'ta.WILLR',
            'roc': 'ta.ROC',
            'mom': 'ta.MOM',
            'highest': 'ta.MAX',
            'lowest': 'ta.MIN',
            'crossover': 'self._crossover',
            'crossunder': 'self._crossunder',
            'cross': 'self._cross'
        }
        
        # Pine Script variable mappings
        self.variable_mappings = {
            'open': 'self.data.open',
            'high': 'self.data.high', 
            'low': 'self.data.low',
            'close': 'self.data.close',
            'volume': 'self.data.volume',
            'hlc3': '(self.data.high + self.data.low + self.data.close) / 3',
            'ohlc4': '(self.data.open + self.data.high + self.data.low + self.data.close) / 4',
            'hl2': '(self.data.high + self.data.low) / 2'
        }
    
    def convert(self, pinescript_code: str) -> Tuple[str, List[ConversionWarning]]:
        """
        Convert Pine Script code to Python backtrader strategy.
        
        Args:
            pinescript_code: The Pine Script code to convert
            
        Returns:
            Tuple of (converted_python_code, warnings)
        """
        self.warnings = []
        
        try:
            # Parse Pine Script
            parsed_strategy = self._parse_pinescript(pinescript_code)
            
            # Generate Python code
            python_code = self._generate_python_strategy(parsed_strategy)
            
            return python_code, self.warnings
            
        except Exception as e:
            self.logger.error(f"Conversion failed: {e}")
            raise ConversionError(f"Failed to convert Pine Script: {e}")
    
    def _parse_pinescript(self, code: str) -> Dict[str, Any]:
        """Parse Pine Script code and extract components."""
        lines = code.split('\n')
        strategy_info = {
            'title': 'Converted Strategy',
            'overlay': False,
            'inputs': [],
            'variables': [],
            'indicators': [],
            'conditions': [],
            'strategy_calls': []
        }
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
                
            # Parse strategy declaration
            if line.startswith('//@version='):
                continue
            elif line.startswith('strategy('):
                strategy_info.update(self._parse_strategy_declaration(line))
            elif line.startswith('input'):
                strategy_info['inputs'].append(self._parse_input(line, i))
            elif '=' in line and not line.startswith('strategy.'):
                strategy_info['variables'].append(self._parse_variable(line, i))
            elif line.startswith('strategy.entry') or line.startswith('strategy.close'):
                strategy_info['strategy_calls'].append(self._parse_strategy_call(line, i))
            elif any(func in line for func in self.function_mappings.keys()):
                strategy_info['indicators'].append(self._parse_indicator(line, i))
        
        return strategy_info
    
    def _parse_strategy_declaration(self, line: str) -> Dict[str, Any]:
        """Parse strategy() declaration."""
        info = {}
        # Extract title
        title_match = re.search(r'title\s*=\s*["\']([^"\']+)["\']', line)
        if title_match:
            info['title'] = title_match.group(1)
        
        # Extract overlay
        overlay_match = re.search(r'overlay\s*=\s*(true|false)', line)
        if overlay_match:
            info['overlay'] = overlay_match.group(1) == 'true'
        
        return info
    
    def _parse_input(self, line: str, line_num: int) -> Dict[str, Any]:
        """Parse input declaration."""
        # Example: length = input(14, "RSI Length")
        match = re.match(r'(\w+)\s*=\s*input\(([^)]+)\)', line)
        if match:
            var_name = match.group(1)
            params = match.group(2).split(',')
            
            input_info = {
                'name': var_name,
                'default_value': params[0].strip(),
                'title': params[1].strip().strip('"\'') if len(params) > 1 else var_name,
                'type': 'int'  # Default type
            }
            
            # Determine type from default value
            try:
                float(input_info['default_value'])
                input_info['type'] = 'float' if '.' in input_info['default_value'] else 'int'
            except ValueError:
                input_info['type'] = 'str'
            
            return input_info
        
        self.warnings.append(ConversionWarning(
            line_num, f"Could not parse input: {line}", line
        ))
        return {}
    
    def _parse_variable(self, line: str, line_num: int) -> Dict[str, Any]:
        """Parse variable assignment."""
        parts = line.split('=', 1)
        if len(parts) == 2:
            var_name = parts[0].strip()
            expression = parts[1].strip()
            
            # Convert Pine Script expression to Python
            converted_expr = self._convert_expression(expression, line_num)
            
            return {
                'name': var_name,
                'expression': converted_expr,
                'original': expression
            }
        
        return {}
    
    def _parse_strategy_call(self, line: str, line_num: int) -> Dict[str, Any]:
        """Parse strategy.entry() or strategy.close() calls."""
        if 'strategy.entry' in line:
            # Example: strategy.entry("Long", strategy.long, when=condition)
            match = re.match(r'strategy\.entry\s*\(\s*["\']([^"\']+)["\'].*when\s*=\s*([^)]+)', line)
            if match:
                return {
                    'type': 'entry',
                    'id': match.group(1),
                    'condition': self._convert_expression(match.group(2), line_num),
                    'direction': 'long' if 'strategy.long' in line else 'short'
                }
        elif 'strategy.close' in line:
            match = re.match(r'strategy\.close\s*\(\s*["\']([^"\']+)["\'].*when\s*=\s*([^)]+)', line)
            if match:
                return {
                    'type': 'close',
                    'id': match.group(1),
                    'condition': self._convert_expression(match.group(2), line_num)
                }
        
        self.warnings.append(ConversionWarning(
            line_num, f"Could not parse strategy call: {line}", line
        ))
        # Return empty dict with default values
        return {
            'type': 'unknown',
            'id': 'unknown',
            'condition': 'False'
        }
    
    def _parse_indicator(self, line: str, line_num: int) -> Dict[str, Any]:
        """Parse indicator calculation."""
        parts = line.split('=', 1)
        if len(parts) == 2:
            var_name = parts[0].strip()
            expression = parts[1].strip()
            
            return {
                'name': var_name,
                'expression': self._convert_expression(expression, line_num),
                'original': expression
            }
        
        # Return empty dict with default values if parsing fails
        return {
            'name': '',
            'expression': '',
            'original': line
        }
    
    def _convert_expression(self, expression: str, line_num: int) -> str:
        """Convert Pine Script expression to Python."""
        converted = expression
        
        # Replace Pine Script functions
        for pine_func, python_func in self.function_mappings.items():
            if pine_func in converted:
                if pine_func in ['sma', 'ema', 'rsi']:
                    # Handle indicators with period parameter
                    pattern = rf'{pine_func}\s*\(\s*([^,]+),\s*(\d+)\s*\)'
                    replacement = rf'{python_func}(self.data.close, timeperiod=\2)'
                    converted = re.sub(pattern, replacement, converted)
                else:
                    converted = converted.replace(pine_func, python_func)
        
        # Replace Pine Script variables
        for pine_var, python_var in self.variable_mappings.items():
            converted = converted.replace(pine_var, python_var)
        
        # Handle array access [1] -> [-1] (Pine Script uses 1 for previous bar)
        converted = re.sub(r'\[(\d+)\]', r'[-\1]', converted)
        
        return converted
    
    def _generate_python_strategy(self, parsed_strategy: Dict[str, Any]) -> str:
        """Generate complete Python backtrader strategy."""
        
        strategy_name = self._to_class_name(parsed_strategy['title'])
        
        code = f'''"""
Generated Strategy: {parsed_strategy['title']}
Converted from Pine Script using TradingView Strategy Generator
"""

import backtrader as bt
import ta
import numpy as np
from typing import Optional

class {strategy_name}(bt.Strategy):
    """
    {parsed_strategy['title']}
    Converted from Pine Script
    """
    
    params = (
'''

        # Add parameters from inputs
        for input_param in parsed_strategy['inputs']:
            default_val = input_param['default_value']
            code += f"        ('{input_param['name']}', {default_val}),  # {input_param['title']}\n"
        
        code += '''    )
    
    def __init__(self):
        # Initialize indicators
'''

        # Add indicator calculations
        for indicator in parsed_strategy['indicators']:
            code += f"        self.{indicator['name']} = {indicator['expression']}\n"
        
        # Add variable calculations
        for variable in parsed_strategy['variables']:
            code += f"        self.{variable['name']} = {variable['expression']}\n"

        code += '''
    def next(self):
        """Execute strategy logic on each bar."""
        # Strategy logic
'''

        # Add strategy calls
        for call in parsed_strategy['strategy_calls']:
            if call['type'] == 'entry':
                if call['direction'] == 'long':
                    code += f'''        if {call['condition']}:
            if not self.position:
                self.buy()  # {call['id']}
'''
                else:
                    code += f'''        if {call['condition']}:
            if not self.position:
                self.sell()  # {call['id']}
'''
            elif call['type'] == 'close':
                code += f'''        if {call['condition']}:
            if self.position:
                self.close()  # {call['id']}
'''

        # Add helper methods
        code += '''
    def _crossover(self, series1, series2):
        """Check if series1 crosses over series2."""
        return series1[0] > series2[0] and series1[-1] <= series2[-1]
    
    def _crossunder(self, series1, series2):
        """Check if series1 crosses under series2."""
        return series1[0] < series2[0] and series1[-1] >= series2[-1]
    
    def _cross(self, series1, series2):
        """Check if series1 crosses series2 in either direction."""
        return self._crossover(series1, series2) or self._crossunder(series1, series2)
    
    def _calculate_macd(self, close, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        exp1 = close.ewm(span=fast).mean()
        exp2 = close.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line, macd - signal_line
    
    def _calculate_bollinger_bands(self, close, period=20, std=2):
        """Calculate Bollinger Bands."""
        sma = close.rolling(window=period).mean()
        std_dev = close.rolling(window=period).std()
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        return upper, sma, lower
    
    def _calculate_stochastic(self, high, low, close, k_period=14, d_period=3):
        """Calculate Stochastic oscillator."""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent
'''

        return code
    
    def _to_class_name(self, title: str) -> str:
        """Convert strategy title to valid Python class name."""
        # Remove special characters and convert to PascalCase
        clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title)
        words = clean_title.split()
        return ''.join(word.capitalize() for word in words) + 'Strategy'

class ConversionError(Exception):
    """Exception raised when Pine Script conversion fails."""
    pass