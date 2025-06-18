#!/usr/bin/env python3
"""
Data Manager for Live Insurance Risk Assessment System
Handles data cleanup, archiving, and optimization to prevent system overload
"""

import os
import json
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any
import threading
import time
from pathlib import Path

class DataManager:
    def __init__(self, data_dir: str = "./live_data_feed", archive_dir: str = "./archived_data"):
        self.data_dir = Path(data_dir)
        self.archive_dir = Path(archive_dir)
        self.max_active_files = 50  # Keep only 50 most recent files active
        self.max_file_age_hours = 24  # Archive files older than 24 hours
        self.cleanup_interval = 300  # Run cleanup every 5 minutes
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.archive_dir.mkdir(exist_ok=True)
        
        # Track running state
        self.running = False
        self.cleanup_thread = None
    
    def start_data_management(self):
        """Start automated data management in background"""
        if self.running:
            return
        
        self.running = True
        self.cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self.cleanup_thread.start()
        print("🗂️ Data management started - will maintain optimal file count")
    
    def stop_data_management(self):
        """Stop automated data management"""
        self.running = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        print("🛑 Data management stopped")
    
    def _cleanup_loop(self):
        """Main cleanup loop running in background"""
        while self.running:
            try:
                self.cleanup_old_files()
                self.maintain_file_limit()
                time.sleep(self.cleanup_interval)
            except Exception as e:
                print(f"🚨 Data cleanup error: {e}")
                time.sleep(60)  # Wait 1 minute before retrying
    
    def cleanup_old_files(self):
        """Archive files older than max_file_age_hours"""
        if not self.data_dir.exists():
            return
        
        cutoff_time = datetime.now() - timedelta(hours=self.max_file_age_hours)
        archived_count = 0
        
        for json_file in self.data_dir.glob("*.json"):
            try:
                # Skip special files
                if json_file.name in ['init.json', 'test_alert.json']:
                    continue
                
                # Check file age
                file_time = datetime.fromtimestamp(json_file.stat().st_mtime)
                if file_time < cutoff_time:
                    self.archive_file(json_file)
                    archived_count += 1
            except Exception as e:
                print(f"⚠️ Error processing {json_file}: {e}")
        
        if archived_count > 0:
            print(f"📦 Archived {archived_count} old files")
    
    def maintain_file_limit(self):
        """Keep only the most recent max_active_files in the active directory"""
        if not self.data_dir.exists():
            return
        
        json_files = list(self.data_dir.glob("*.json"))
        
        # Filter out special files
        active_files = [f for f in json_files if f.name not in ['init.json', 'test_alert.json']]
        
        if len(active_files) <= self.max_active_files:
            return
        
        # Sort by modification time (newest first)
        active_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Archive excess files
        files_to_archive = active_files[self.max_active_files:]
        archived_count = 0
        
        for file_path in files_to_archive:
            try:
                self.archive_file(file_path)
                archived_count += 1
            except Exception as e:
                print(f"⚠️ Error archiving {file_path}: {e}")
        
        if archived_count > 0:
            print(f"📦 Archived {archived_count} files to maintain limit of {self.max_active_files}")
    
    def archive_file(self, file_path: Path):
        """Archive a single file by moving it to the archive directory"""
        if not file_path.exists():
            return
        
        # Create date-based subdirectory in archive
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                timestamp = data.get('timestamp', '')
                if timestamp:
                    date_str = timestamp[:10]  # YYYY-MM-DD
                else:
                    date_str = datetime.now().strftime('%Y-%m-%d')
        except:
            date_str = datetime.now().strftime('%Y-%m-%d')
        
        archive_subdir = self.archive_dir / date_str
        archive_subdir.mkdir(exist_ok=True)
        
        # Move file to archive
        archive_path = archive_subdir / file_path.name
        shutil.move(str(file_path), str(archive_path))
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get statistics about the data management system"""
        active_files = len(list(self.data_dir.glob("*.json"))) if self.data_dir.exists() else 0
        
        archive_files = 0
        archive_size_mb = 0
        if self.archive_dir.exists():
            for archive_file in self.archive_dir.rglob("*.json"):
                archive_files += 1
                archive_size_mb += archive_file.stat().st_size / (1024 * 1024)
        
        # Analyze data sources in active files
        sources = {}
        types = {}
        if self.data_dir.exists():
            for json_file in self.data_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        source = data.get('source', 'unknown')
                        data_type = data.get('type', 'unknown')
                        sources[source] = sources.get(source, 0) + 1
                        types[data_type] = types.get(data_type, 0) + 1
                except:
                    pass
        
        return {
            'active_files': active_files,
            'archived_files': archive_files,
            'archive_size_mb': round(archive_size_mb, 2),
            'max_active_files': self.max_active_files,
            'max_file_age_hours': self.max_file_age_hours,
            'data_sources': sources,
            'data_types': types,
            'management_active': self.running
        }
    
    def force_cleanup(self):
        """Force immediate cleanup - useful for manual maintenance"""
        print("🧹 Starting forced cleanup...")
        initial_count = len(list(self.data_dir.glob("*.json"))) if self.data_dir.exists() else 0
        
        self.cleanup_old_files()
        self.maintain_file_limit()
        
        final_count = len(list(self.data_dir.glob("*.json"))) if self.data_dir.exists() else 0
        removed_count = initial_count - final_count
        
        print(f"✅ Cleanup complete: {removed_count} files archived ({final_count} remaining)")
        return {
            'initial_files': initial_count,
            'final_files': final_count,
            'archived_files': removed_count
        }
    
    def get_recent_data_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of data from the last N hours"""
        if not self.data_dir.exists():
            return {'error': 'Data directory not found'}
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_files = []
        
        for json_file in self.data_dir.glob("*.json"):
            try:
                file_time = datetime.fromtimestamp(json_file.stat().st_mtime)
                if file_time >= cutoff_time:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        recent_files.append({
                            'file': json_file.name,
                            'timestamp': data.get('timestamp', ''),
                            'source': data.get('source', 'unknown'),
                            'type': data.get('type', 'unknown'),
                            'location': data.get('location', ''),
                            'age_minutes': int((datetime.now() - file_time).total_seconds() / 60)
                        })
            except Exception as e:
                continue
        
        # Sort by timestamp
        recent_files.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            'hours_analyzed': hours,
            'total_files': len(recent_files),
            'files': recent_files[:20],  # Return latest 20
            'sources': list(set(f['source'] for f in recent_files)),
            'types': list(set(f['type'] for f in recent_files))
        }

if __name__ == "__main__":
    # Test the data manager
    manager = DataManager()
    
    print("🗂️ Data Manager Test")
    print("=" * 40)
    
    # Get current stats
    stats = manager.get_system_stats()
    print(f"Current files: {stats['active_files']}")
    print(f"Archived files: {stats['archived_files']}")
    print(f"Archive size: {stats['archive_size_mb']} MB")
    
    # Show recent data
    recent = manager.get_recent_data_summary(hours=2)
    print(f"\nRecent files (last 2 hours): {recent['total_files']}")
    print(f"Sources: {', '.join(recent['sources'])}")
    print(f"Types: {', '.join(recent['types'])}")
    
    # Demonstrate forced cleanup
    cleanup_result = manager.force_cleanup()
    print(f"\nCleanup result: {cleanup_result}")
