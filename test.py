#!/usr/bin/env python3
"""
Smart Traffic Light Analyzer
Analyzes MP4 traffic footage and provides recommendations for improvements
"""

import cv2
import numpy as np
from datetime import datetime
import json
from collections import defaultdict

class TrafficAnalyzer:
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Analysis metrics
        self.vehicle_count = 0
        self.pedestrian_count = 0
        self.congestion_levels = []
        self.stop_events = []
        self.flow_rates = []
        
        # Background subtractor for motion detection
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
    def detect_objects(self, frame):
        """Detect moving objects using background subtraction"""
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows and noise
        _, fg_mask = cv2.threshold(fg_mask, 250, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'type': 'vehicle' if area > 2000 else 'pedestrian'
                })
        
        return detections, fg_mask
    
    def calculate_congestion(self, detections):
        """Calculate congestion level based on detections"""
        if not detections:
            return 0.0
        
        total_area = sum(d['area'] for d in detections)
        frame_area = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        congestion = min(total_area / (frame_area * 0.3), 1.0)
        return congestion
    
    def analyze_video(self, sample_rate=5):
        """Analyze video and collect metrics"""
        print(f"Analyzing video: {self.video_path}")
        print(f"Duration: {self.duration:.1f}s, FPS: {self.fps}, Total Frames: {self.total_frames}")
        print("-" * 60)
        
        frame_count = 0
        vehicles_per_frame = []
        pedestrians_per_frame = []
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames to speed up processing
            if frame_count % sample_rate != 0:
                continue
            
            # Detect objects
            detections, fg_mask = self.detect_objects(frame)
            
            # Count vehicles and pedestrians
            vehicles = [d for d in detections if d['type'] == 'vehicle']
            pedestrians = [d for d in detections if d['type'] == 'pedestrian']
            
            vehicles_per_frame.append(len(vehicles))
            pedestrians_per_frame.append(len(pedestrians))
            
            # Calculate congestion
            congestion = self.calculate_congestion(detections)
            self.congestion_levels.append(congestion)
            
            # Progress indicator
            progress = (frame_count / self.total_frames) * 100
            if frame_count % 100 == 0:
                print(f"Progress: {progress:.1f}% - Vehicles: {len(vehicles)}, Pedestrians: {len(pedestrians)}, Congestion: {congestion:.2f}")
        
        self.cap.release()
        
        # Calculate statistics
        self.vehicle_count = int(np.mean(vehicles_per_frame)) if vehicles_per_frame else 0
        self.pedestrian_count = int(np.mean(pedestrians_per_frame)) if pedestrians_per_frame else 0
        self.avg_congestion = np.mean(self.congestion_levels) if self.congestion_levels else 0
        self.max_congestion = np.max(self.congestion_levels) if self.congestion_levels else 0
        
        print("\n" + "=" * 60)
        print("Analysis Complete!")
        print("=" * 60)
    
    def generate_recommendations(self):
        """Generate recommendations based on analysis"""
        issues = []
        recommendations = []
        
        # Analyze congestion
        if self.avg_congestion > 0.7:
            issues.append({
                'type': 'High Congestion',
                'severity': 'HIGH',
                'description': f'Average congestion level at {self.avg_congestion*100:.1f}% - significantly above normal'
            })
            recommendations.append({
                'category': 'Traffic Signals',
                'priority': 'HIGH',
                'suggestion': 'Install adaptive traffic light system with real-time density detection',
                'impact': 'Could reduce congestion by 25-30%',
                'cost': 'Medium ($50k-$100k)'
            })
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'HIGH',
                'suggestion': 'Add dedicated turn lanes to reduce intersection blocking',
                'impact': 'Improve flow rate by 15-20%',
                'cost': 'High ($200k+)'
            })
        elif self.avg_congestion > 0.4:
            issues.append({
                'type': 'Moderate Congestion',
                'severity': 'MEDIUM',
                'description': f'Average congestion level at {self.avg_congestion*100:.1f}% - approaching concerning levels'
            })
            recommendations.append({
                'category': 'Traffic Signals',
                'priority': 'MEDIUM',
                'suggestion': 'Optimize traffic light timing based on peak hours',
                'impact': 'Could reduce delays by 10-15%',
                'cost': 'Low ($5k-$10k)'
            })
        
        # Analyze vehicle count
        if self.vehicle_count > 30:
            issues.append({
                'type': 'High Vehicle Volume',
                'severity': 'MEDIUM',
                'description': f'Average of {self.vehicle_count} vehicles per frame detected'
            })
            recommendations.append({
                'category': 'Traffic Management',
                'priority': 'MEDIUM',
                'suggestion': 'Implement variable speed limits during peak hours',
                'impact': 'Smooth traffic flow, reduce accidents by 10%',
                'cost': 'Low ($10k-$20k)'
            })
        
        # Analyze pedestrian safety
        if self.pedestrian_count > 5:
            issues.append({
                'type': 'Pedestrian Safety Concern',
                'severity': 'MEDIUM',
                'description': f'Average of {self.pedestrian_count} pedestrians detected - high foot traffic'
            })
            recommendations.append({
                'category': 'Safety',
                'priority': 'HIGH',
                'suggestion': 'Install dedicated pedestrian crossing signals with countdown timers',
                'impact': 'Reduce pedestrian accidents by 30-40%',
                'cost': 'Medium ($30k-$50k)'
            })
            recommendations.append({
                'category': 'Infrastructure',
                'priority': 'MEDIUM',
                'suggestion': 'Add raised crosswalks or pedestrian islands for safer crossing',
                'impact': 'Improve pedestrian safety and visibility',
                'cost': 'Medium ($40k-$80k)'
            })
        
        # Peak congestion analysis
        if self.max_congestion > 0.85:
            recommendations.append({
                'category': 'Monitoring',
                'priority': 'HIGH',
                'suggestion': 'Install real-time traffic cameras and AI monitoring system',
                'impact': 'Enable proactive traffic management and incident response',
                'cost': 'High ($100k-$200k)'
            })
        
        # General recommendations
        if len(issues) > 0:
            recommendations.append({
                'category': 'Smart City',
                'priority': 'MEDIUM',
                'suggestion': 'Deploy IoT sensors for continuous traffic monitoring',
                'impact': 'Collect long-term data for better planning',
                'cost': 'Medium ($50k-$100k annually)'
            })
        
        return issues, recommendations
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        issues, recommendations = self.generate_recommendations()
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'video_info': {
                'path': self.video_path,
                'duration_seconds': round(self.duration, 2),
                'fps': self.fps,
                'total_frames': self.total_frames
            },
            'metrics': {
                'avg_vehicles_per_frame': self.vehicle_count,
                'avg_pedestrians_per_frame': self.pedestrian_count,
                'avg_congestion_level': round(self.avg_congestion, 3),
                'max_congestion_level': round(self.max_congestion, 3),
                'congestion_rating': self.get_congestion_rating()
            },
            'issues_detected': issues,
            'recommendations': recommendations,
            'summary': self.generate_summary(issues, recommendations)
        }
        
        return report
    
    def get_congestion_rating(self):
        """Get human-readable congestion rating"""
        if self.avg_congestion < 0.3:
            return 'LOW - Traffic flowing smoothly'
        elif self.avg_congestion < 0.5:
            return 'MODERATE - Some delays expected'
        elif self.avg_congestion < 0.7:
            return 'HIGH - Significant congestion present'
        else:
            return 'CRITICAL - Severe congestion, immediate action needed'
    
    def generate_summary(self, issues, recommendations):
        """Generate executive summary"""
        if not issues:
            return "Traffic conditions are within acceptable parameters. No immediate action required."
        
        summary = f"Analysis identified {len(issues)} issue(s) requiring attention. "
        summary += f"Traffic congestion level is {self.get_congestion_rating()}. "
        summary += f"Recommended {len(recommendations)} improvement(s) prioritized by impact and urgency."
        
        return summary
    
    def print_report(self):
        """Print formatted report to console"""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("SMART TRAFFIC ANALYSIS REPORT".center(80))
        print("=" * 80)
        
        print(f"\n📅 Analysis Date: {report['analysis_date']}")
        print(f"🎥 Video: {report['video_info']['path']}")
        print(f"⏱️  Duration: {report['video_info']['duration_seconds']}s")
        
        print("\n" + "-" * 80)
        print("TRAFFIC METRICS".center(80))
        print("-" * 80)
        
        metrics = report['metrics']
        print(f"🚗 Average Vehicles per Frame: {metrics['avg_vehicles_per_frame']}")
        print(f"🚶 Average Pedestrians per Frame: {metrics['avg_pedestrians_per_frame']}")
        print(f"📊 Average Congestion Level: {metrics['avg_congestion_level']*100:.1f}%")
        print(f"📈 Peak Congestion Level: {metrics['max_congestion_level']*100:.1f}%")
        print(f"⚠️  Congestion Rating: {metrics['congestion_rating']}")
        
        if report['issues_detected']:
            print("\n" + "-" * 80)
            print("ISSUES DETECTED".center(80))
            print("-" * 80)
            
            for i, issue in enumerate(report['issues_detected'], 1):
                print(f"\n{i}. [{issue['severity']}] {issue['type']}")
                print(f"   {issue['description']}")
        
        if report['recommendations']:
            print("\n" + "-" * 80)
            print("RECOMMENDATIONS".center(80))
            print("-" * 80)
            
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"\n{i}. [{rec['priority']}] {rec['category']}")
                print(f"   💡 {rec['suggestion']}")
                print(f"   📊 Impact: {rec['impact']}")
                print(f"   💰 Cost: {rec['cost']}")
        
        print("\n" + "-" * 80)
        print("SUMMARY".center(80))
        print("-" * 80)
        print(f"\n{report['summary']}")
        
        print("\n" + "=" * 80)
        print("END OF REPORT".center(80))
        print("=" * 80)
    
    def save_report(self, output_path='traffic_analysis_report.json'):
        """Save report to JSON file"""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {output_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("SMART TRAFFIC LIGHT ANALYZER".center(80))
    print("=" * 80)
    
    # Get video path from user
    video_path = input("\n🎥 Enter the path to your MP4 traffic video: ").strip()
    
    # Check if file exists
    try:
        analyzer = TrafficAnalyzer(video_path)
    except Exception as e:
        print(f"\n❌ Error: Could not open video file. {str(e)}")
        return
    
    # Analyze video
    print("\n🔍 Starting analysis...\n")
    analyzer.analyze_video(sample_rate=5)
    
    # Print report
    analyzer.print_report()
    
    # Save report
    save_option = input("\n💾 Save report to JSON file? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = input("Enter filename (default: traffic_analysis_report.json): ").strip()
        if not filename:
            filename = 'traffic_analysis_report.json'
        analyzer.save_report(filename)
    
    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()