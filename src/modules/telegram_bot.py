"""
Telegram Bot Module
Handles Telegram integration for sending trading signals and receiving commands.
"""

import logging
import asyncio
from datetime import datetime
from typing import Optional, List, Dict, Any
import telegram
from telegram.ext import Application, CommandHandler, MessageHandler, filters

from core.database import DatabaseManager, Signal, Strategy


class TelegramSignalBot:
    """Telegram bot for trading signal notifications."""
    
    def __init__(self, bot_token: Optional[str], chat_id: Optional[str] = None):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.logger = logging.getLogger(__name__)
        self.bot = None
        self.application = None
        
        if bot_token:
            try:
                self.bot = telegram.Bot(token=bot_token)
            except Exception as e:
                self.logger.error(f"Failed to initialize Telegram bot: {e}")
    
    def format_signal_message(self, signal: Signal) -> str:
        """Format a trading signal for Telegram message."""
        strategy_name = signal.strategy.name if signal.strategy else "Unknown Strategy"
        
        # Basic signal info
        message = f"🚨 **{signal.signal_type} SIGNAL** 🚨\n\n"
        message += f"📊 **Strategy:** {strategy_name}\n"
        message += f"💰 **Symbol:** {signal.symbol}\n"
        message += f"💵 **Price:** ${signal.price:.4f}\n"
        
        # Add take profit and stop loss if available
        if signal.take_profit:
            message += f"🎯 **TP:** ${signal.take_profit:.4f}\n"
        
        if signal.stop_loss:
            message += f"🛑 **SL:** ${signal.stop_loss:.4f}\n"
        
        # Add quantity and timestamp
        if signal.quantity:
            message += f"📦 **Quantity:** {signal.quantity}\n"
        
        message += f"⏰ **Time:** {signal.created_at.strftime('%Y-%m-%d %H:%M:%S') if signal.created_at else 'N/A'}\n"
        
        # Add formatted signal line for easy copy-paste
        tp_text = f" TP:{signal.take_profit:.0f}" if signal.take_profit else ""
        sl_text = f" SL:{signal.stop_loss:.0f}" if signal.stop_loss else ""
        message += f"\n`{signal.symbol} {signal.signal_type} {signal.price:.0f}{tp_text}{sl_text}`"
        
        return message
    
    def format_signal_close_message(self, signal: Signal) -> str:
        """Format a signal close message for Telegram."""
        strategy_name = signal.strategy.name if signal.strategy else "Unknown Strategy"
        
        # Determine if profitable
        profit_emoji = "💚" if signal.is_profitable else "❌"
        pnl_text = f"+${signal.pnl:.2f}" if signal.pnl and signal.pnl > 0 else f"-${abs(signal.pnl or 0):.2f}"
        
        message = f"{profit_emoji} **SIGNAL CLOSED** {profit_emoji}\n\n"
        message += f"📊 **Strategy:** {strategy_name}\n"
        message += f"💰 **Symbol:** {signal.symbol}\n"
        message += f"📈 **Entry:** ${signal.price:.4f}\n"
        message += f"📉 **Exit:** ${signal.close_price:.4f}\n"
        message += f"💵 **P&L:** {pnl_text}\n"
        
        if signal.closed_at:
            message += f"⏰ **Closed:** {signal.closed_at.strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        return message
    
    async def send_signal(self, signal: Signal) -> bool:
        """Send a trading signal to Telegram."""
        if not self.bot or not self.chat_id:
            self.logger.warning("Telegram bot not configured")
            return False
        
        try:
            message = self.format_signal_message(signal)
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            # Update signal to mark as sent
            signal.is_sent_to_telegram = True
            
            self.logger.info(f"Sent signal {signal.id} to Telegram")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send signal to Telegram: {e}")
            return False
    
    async def send_signal_close(self, signal: Signal) -> bool:
        """Send a signal close notification to Telegram."""
        if not self.bot or not self.chat_id:
            return False
        
        try:
            message = self.format_signal_close_message(signal)
            
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            
            self.logger.info(f"Sent signal close {signal.id} to Telegram")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send signal close to Telegram: {e}")
            return False
    
    async def send_custom_message(self, message: str) -> bool:
        """Send a custom message to Telegram."""
        if not self.bot or not self.chat_id:
            return False
        
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='Markdown'
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to send custom message: {e}")
            return False
    
    def setup_bot_commands(self, db_manager: DatabaseManager):
        """Setup bot commands for user interaction."""
        if not self.bot_token:
            return
        
        self.application = Application.builder().token(self.bot_token).build()
        
        # Add command handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(CommandHandler("status", lambda update, context: self._status_command(update, context, db_manager)))
        self.application.add_handler(CommandHandler("signals", lambda update, context: self._signals_command(update, context, db_manager)))
        self.application.add_handler(CommandHandler("performance", lambda update, context: self._performance_command(update, context, db_manager)))
        self.application.add_handler(CommandHandler("help", self._help_command))
    
    async def _start_command(self, update, context):
        """Handle /start command."""
        welcome_message = """
🚀 **Trading Signal Bot** 🚀

Welcome! This bot will send you live trading signals from your strategies.

Available commands:
/status - Check bot status
/signals - View open signals
/performance - View strategy performance
/help - Show this help message
        """
        await update.message.reply_text(welcome_message, parse_mode='Markdown')
    
    async def _status_command(self, update, context, db_manager: DatabaseManager):
        """Handle /status command."""
        try:
            # Get signal statistics
            open_signals = db_manager.get_open_signals()
            strategies = db_manager.get_all_strategies()
            active_strategies = [s for s in strategies if s.is_active]
            
            message = f"📊 **Bot Status**\n\n"
            message += f"🔄 **Active Strategies:** {len(active_strategies)}\n"
            message += f"📡 **Open Signals:** {len(open_signals)}\n"
            message += f"⏰ **Last Update:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"Error getting status: {e}")
    
    async def _signals_command(self, update, context, db_manager: DatabaseManager):
        """Handle /signals command."""
        try:
            open_signals = db_manager.get_open_signals()
            
            if not open_signals:
                await update.message.reply_text("📭 No open signals currently.")
                return
            
            message = f"📡 **Open Signals ({len(open_signals)})**\n\n"
            
            for i, signal in enumerate(open_signals[:10], 1):  # Limit to 10 signals
                strategy_name = signal.strategy.name if signal.strategy else "Unknown"
                age_hours = (datetime.now() - signal.created_at).total_seconds() / 3600 if signal.created_at else 0
                
                message += f"{i}. **{signal.symbol}** - {signal.signal_type}\n"
                message += f"   💰 ${signal.price:.4f} | 🕐 {age_hours:.1f}h ago\n"
                message += f"   📊 {strategy_name}\n\n"
            
            if len(open_signals) > 10:
                message += f"... and {len(open_signals) - 10} more signals"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"Error getting signals: {e}")
    
    async def _performance_command(self, update, context, db_manager: DatabaseManager):
        """Handle /performance command."""
        try:
            strategies = db_manager.get_all_strategies()
            
            if not strategies:
                await update.message.reply_text("📭 No strategies found.")
                return
            
            message = "📈 **Strategy Performance**\n\n"
            
            for strategy in strategies[:5]:  # Limit to 5 strategies
                perf = db_manager.get_strategy_performance(strategy.id)
                
                if perf['total_signals'] > 0:
                    message += f"**{strategy.name}**\n"
                    message += f"📊 Signals: {perf['total_signals']}\n"
                    message += f"🎯 Win Rate: {perf['win_rate']:.1%}\n"
                    message += f"💰 Total P&L: ${perf['total_pnl']:.2f}\n\n"
            
            await update.message.reply_text(message, parse_mode='Markdown')
            
        except Exception as e:
            await update.message.reply_text(f"Error getting performance: {e}")
    
    async def _help_command(self, update, context):
        """Handle /help command."""
        help_message = """
🤖 **Trading Signal Bot Commands**

/start - Welcome message
/status - Check bot and signal status
/signals - View current open signals
/performance - View strategy performance stats
/help - Show this help message

The bot automatically sends notifications when:
• New signals are generated
• Signals are closed (profit/loss)

Happy trading! 📈
        """
        await update.message.reply_text(help_message, parse_mode='Markdown')
    
    def start_polling(self):
        """Start the bot polling for commands."""
        if self.application:
            self.application.run_polling()
    
    async def test_connection(self) -> bool:
        """Test the Telegram bot connection."""
        if not self.bot:
            return False
        
        try:
            me = await self.bot.get_me()
            self.logger.info(f"Telegram bot connected: @{me.username}")
            return True
        except Exception as e:
            self.logger.error(f"Telegram bot connection failed: {e}")
            return False


class TelegramSignalManager:
    """Manages Telegram integration for the signal system."""
    
    def __init__(self, db_manager: DatabaseManager, bot_token: str = None, chat_id: str = None):
        self.db = db_manager
        self.bot = TelegramSignalBot(bot_token, chat_id)
        self.logger = logging.getLogger(__name__)
        
        # Setup bot commands if token is provided
        if bot_token:
            self.bot.setup_bot_commands(db_manager)
    
    async def process_new_signal(self, signal: Signal) -> bool:
        """Process a new signal for Telegram notification."""
        try:
            # Send signal to Telegram
            success = await self.bot.send_signal(signal)
            
            if success:
                # Update signal in database
                session = self.db.get_session()
                try:
                    session.merge(signal)
                    session.commit()
                finally:
                    session.close()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error processing new signal for Telegram: {e}")
            return False
    
    async def process_signal_close(self, signal: Signal) -> bool:
        """Process a signal close for Telegram notification."""
        try:
            return await self.bot.send_signal_close(signal)
        except Exception as e:
            self.logger.error(f"Error processing signal close for Telegram: {e}")
            return False
    
    async def send_daily_summary(self) -> bool:
        """Send daily performance summary."""
        try:
            # Get today's signals
            today = datetime.now().date()
            session = self.db.get_session()
            
            try:
                from sqlalchemy import func, and_
                today_signals = session.query(Signal).filter(
                    func.date(Signal.created_at) == today
                ).all()
                
                closed_signals = [s for s in today_signals if s.status == 'CLOSED']
                profitable_signals = [s for s in closed_signals if s.is_profitable]
                
                if not today_signals:
                    message = "📊 **Daily Summary**\n\nNo signals generated today."
                else:
                    total_pnl = sum(s.pnl or 0 for s in closed_signals)
                    win_rate = len(profitable_signals) / len(closed_signals) if closed_signals else 0
                    
                    message = f"📊 **Daily Summary - {today.strftime('%Y-%m-%d')}**\n\n"
                    message += f"📡 Total Signals: {len(today_signals)}\n"
                    message += f"✅ Closed Signals: {len(closed_signals)}\n"
                    message += f"🎯 Win Rate: {win_rate:.1%}\n"
                    message += f"💰 Total P&L: ${total_pnl:.2f}\n"
                
                return await self.bot.send_custom_message(message)
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error sending daily summary: {e}")
            return False