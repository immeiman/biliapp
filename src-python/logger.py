"""
logger.py

Log bilirubin predictions and capture metadata to CSV/SQLite.
"""

import csv
import logging
import sqlite3
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

_log = logging.getLogger(__name__)


class PredictionLogger:
    """
    Log prediction results to CSV and/or SQLite database.
    """

    def __init__(self, log_dir: str = "logs", use_csv: bool = True, use_sqlite: bool = False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.use_csv = use_csv
        self.use_sqlite = use_sqlite
        self.last_write_error: Optional[str] = None

        # CSV file path
        self.csv_path = self.log_dir / "predictions.csv"
        
        # SQLite database path
        if self.use_sqlite:
            self.db_path = self.log_dir / "predictions.db"
            self._init_db()
        else:
            self.db_path = None

        # Initialize CSV header if needed
        if self.use_csv and not self.csv_path.exists():
            self._init_csv()

    _CSV_FIELDNAMES = [
        'timestamp', 'image_filename', 'image_path',
        'bilirubin_prediction', 'preprocessing_mode',
        'quality_label', 'quality_score', 'success',
        'error_message', 'model_version', 'notes',
    ]

    def _init_csv(self):
        """Create CSV file with header."""
        try:
            with open(self.csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDNAMES)
                writer.writeheader()
                f.flush()
        except Exception as e:
            _log.warning("Error initializing CSV: %s", e)

    def _init_db(self):
        """Create SQLite database with predictions table."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    image_filename TEXT,
                    image_path TEXT,
                    bilirubin_prediction REAL,
                    preprocessing_mode TEXT,
                    quality_label TEXT,
                    quality_score INTEGER,
                    success BOOLEAN,
                    error_message TEXT,
                    model_version TEXT,
                    notes TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
        except Exception as e:
            _log.warning("Error initializing SQLite: %s", e)

    def log_prediction(self, **kwargs) -> bool:
        """
        Log a prediction result.
        
        Args:
            timestamp: datetime object (default: now)
            image_filename: Original filename
            image_path: Full path to saved image
            bilirubin_prediction: Predicted value (float)
            preprocessing_mode: 'raw_aligned', 'white_balance_only', 'wb_plus_palette'
            quality_label: 'high', 'medium', 'low'
            quality_score: 0-100
            success: True/False
            error_message: Error string if success=False
            model_version: Model identifier
            notes: Additional notes
        
        Returns:
            True if logged successfully
        """
        try:
            self.last_write_error = None
            timestamp = kwargs.get('timestamp') or datetime.now()

            record = {
                'timestamp': timestamp.isoformat(),
                'image_filename': kwargs.get('image_filename', ''),
                'image_path': kwargs.get('image_path', ''),
                'bilirubin_prediction': kwargs.get('bilirubin_prediction'),
                'preprocessing_mode': kwargs.get('preprocessing_mode', ''),
                'quality_label': kwargs.get('quality_label', ''),
                'quality_score': kwargs.get('quality_score', 0),
                'success': kwargs.get('success', False),
                'error_message': kwargs.get('error_message', ''),
                'model_version': kwargs.get('model_version', ''),
                'notes': kwargs.get('notes', ''),
            }

            if self.use_csv:
                # Re-create header if file was deleted after startup
                if not self.csv_path.exists():
                    self._init_csv()
                with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=self._CSV_FIELDNAMES)
                    writer.writerow(record)
                    f.flush()

            if self.use_sqlite:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO predictions (
                        timestamp, image_filename, image_path, bilirubin_prediction,
                        preprocessing_mode, quality_label, quality_score, success,
                        error_message, model_version, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record['timestamp'], record['image_filename'], record['image_path'],
                    record['bilirubin_prediction'], record['preprocessing_mode'],
                    record['quality_label'], record['quality_score'], record['success'],
                    record['error_message'], record['model_version'], record['notes'],
                ))
                conn.commit()
                conn.close()

            return True

        except Exception as e:
            self.last_write_error = str(e)
            _log.warning("Error logging prediction: %s", e)
            return False

    def get_last_predictions(self, num: int = 10) -> list:
        """Get last N predictions from CSV."""
        try:
            if not self.csv_path.exists():
                return []

            records = []
            with open(self.csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    records.append(row)

            return records[-num:]

        except Exception:
            return []

    def get_statistics(self) -> Dict[str, Any]:
        """Get basic statistics from logged predictions."""
        try:
            if not self.csv_path.exists():
                return {
                    'total_predictions': 0,
                    'successful': 0,
                    'failed': 0,
                    'mean_bilirubin': None,
                    'quality_distribution': {}
                }

            import pandas as pd
            df = pd.read_csv(self.csv_path)
            
            if 'success' in df.columns:
                successful = df[df['success'].astype(str).str.strip().str.lower() == 'true']
            else:
                successful = pd.DataFrame()
            
            stats = {
                'total_predictions': len(df),
                'successful': len(successful),
                'failed': len(df) - len(successful),
                'mean_bilirubin': float(successful['bilirubin_prediction'].mean()) if len(successful) > 0 else None,
                'quality_distribution': {}
            }
            
            if 'quality_label' in df.columns and len(successful) > 0:
                stats['quality_distribution'] = successful['quality_label'].value_counts().to_dict()

            return stats

        except Exception:
            return {'total_predictions': 0, 'error': 'Could not compute statistics'}

    def export_json(self, output_path: str) -> bool:
        """Export CSV logs to JSON format."""
        try:
            import pandas as pd
            import json
            
            if not self.csv_path.exists():
                return False

            df = pd.read_csv(self.csv_path)
            df.to_json(output_path, orient='records', indent=2, date_format='iso')
            return True

        except Exception:
            return False
