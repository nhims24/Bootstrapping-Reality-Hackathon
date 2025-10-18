#!/usr/bin/env python3
"""
Smart Traffic Light Analyzer with OpenAI Vision API
Analyzes MP4 traffic footage using AI for accurate detection and recommendations
"""

import cv2
import numpy as np
from datetime import datetime
import json
import base64
import os
from collections import defaultdict
from openai import OpenAI

class TrafficAnalyzer:
    def __init__(self, video_path, openai_api_key=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration = self.total_frames / self.fps if self.fps > 0 else 0
        
        # Initialize OpenAI
        self.openai_api_key = openai_api_key or os.getenv('OPENAI_API_KEY')
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass it as parameter.")
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # Analysis storage
        self.frame_analyses = []
        self.vehicle_count = 0
        self.pedestrian_count = 0
        self.congestion_levels = []
        
    def encode_frame_to_base64(self, frame):
        """Encode frame to base64 for OpenAI API"""
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return base64.b64encode(buffer).decode('utf-8')
    
    def analyze_frame_with_ai(self, frame, frame_number, timestamp):
        """Use OpenAI Vision API to analyze traffic frame"""
        try:
            base64_image = self.encode_frame_to_base64(frame)
            
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": """You are a traffic engineering expert. Analyze this traffic scene in detail and provide a comprehensive assessment.

Count and identify:
1. Total number of vehicles (cars, trucks, buses, motorcycles) - be precise
2. Total number of pedestrians visible
3. Traffic density level (0-100% scale)

Assess traffic conditions:
4. Current traffic flow status (flowing, slow, congested, gridlocked)
5. Lane usage and distribution
6. Any bottlenecks or problem areas visible

Safety observations:
7. Any unsafe behaviors (jaywalking, aggressive driving, running red lights)
8. Visibility issues (poor lighting, obstructions, faded markings)
9. Pedestrian safety concerns
10. Accident risk factors

Infrastructure assessment:
11. Road condition and quality
12. Traffic signal visibility and placement
13. Road marking quality
14. Signage adequacy
15. Pedestrian crossing facilities

Specific recommendations:
16. Immediate safety improvements needed
17. Traffic flow optimization suggestions
18. Infrastructure upgrades required
19. Signal timing adjustments needed

Provide response in JSON format:
{
  "vehicle_count": number,
  "vehicle_types": {"cars": X, "trucks": X, "buses": X, "motorcycles": X},
  "pedestrian_count": number,
  "congestion_percent": number (0-100),
  "traffic_flow": "flowing/slow/congested/gridlocked",
  "lane_usage": "description",
  "safety_issues": ["issue1", "issue2"],
  "visibility_problems": ["problem1", "problem2"],
  "infrastructure_issues": ["issue1", "issue2"],
  "immediate_actions": ["action1", "action2"],
  "recommendations": ["rec1", "rec2"],
  "risk_level": "low/medium/high/critical",
  "notes": "additional observations"
}"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=1500,
                temperature=0.3
            )
            
            analysis_text = response.choices[0].message.content
            
            # Extract JSON from response
            try:
                # Try to find JSON in the response
                json_start = analysis_text.find('{')
                json_end = analysis_text.rfind('}') + 1
                if json_start != -1 and json_end > json_start:
                    analysis_json = json.loads(analysis_text[json_start:json_end])
                else:
                    analysis_json = json.loads(analysis_text)
                
                return {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'analysis': analysis_json,
                    'raw_response': analysis_text,
                    'success': True
                }
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parse error for frame {frame_number}: {e}")
                print(f"Response: {analysis_text[:200]}...")
                return {
                    'frame_number': frame_number,
                    'timestamp': timestamp,
                    'analysis': None,
                    'raw_response': analysis_text,
                    'success': False
                }
                
        except Exception as e:
            print(f"❌ AI analysis failed for frame {frame_number}: {str(e)}")
            return None
    
    def analyze_video(self, sample_interval=30):
        """
        Analyze video by sampling frames at regular intervals
        sample_interval: analyze every Nth frame (default 30 = ~1 per second for 30fps video)
        """
        print(f"Analyzing video: {self.video_path}")
        print(f"Duration: {self.duration:.1f}s, FPS: {self.fps}, Total Frames: {self.total_frames}")
        print(f"Sample Interval: Every {sample_interval} frames (~{sample_interval/self.fps:.1f}s)")
        print("-" * 80)
        
        frame_count = 0
        analyzed_count = 0
        
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Sample frames at intervals
            if frame_count % sample_interval == 0:
                timestamp = frame_count / self.fps
                print(f"\n🔍 Analyzing frame {frame_count}/{self.total_frames} (Time: {timestamp:.1f}s)")
                
                result = self.analyze_frame_with_ai(frame, frame_count, timestamp)
                
                if result and result['success']:
                    self.frame_analyses.append(result)
                    analyzed_count += 1
                    
                    # Show quick summary
                    analysis = result['analysis']
                    print(f"   Vehicles: {analysis.get('vehicle_count', 'N/A')}, "
                          f"Pedestrians: {analysis.get('pedestrian_count', 'N/A')}, "
                          f"Congestion: {analysis.get('congestion_percent', 'N/A')}%, "
                          f"Flow: {analysis.get('traffic_flow', 'N/A')}")
                    print(f"   Risk Level: {analysis.get('risk_level', 'N/A').upper()}")
                
                progress = (frame_count / self.total_frames) * 100
                print(f"   Progress: {progress:.1f}%")
            
            # Show progress every 100 frames
            elif frame_count % 100 == 0:
                progress = (frame_count / self.total_frames) * 100
                print(f"Progress: {progress:.1f}% (Analyzed {analyzed_count} frames so far)")
        
        self.cap.release()
        
        print("\n" + "=" * 80)
        print(f"Analysis Complete! Analyzed {analyzed_count} frames with AI")
        print("=" * 80)
        
        # Calculate aggregated metrics
        self._calculate_metrics()
    
    def _calculate_metrics(self):
        """Calculate aggregated metrics from AI analyses"""
        if not self.frame_analyses:
            return
        
        vehicle_counts = []
        pedestrian_counts = []
        congestion_levels = []
        
        for result in self.frame_analyses:
            if result['analysis']:
                analysis = result['analysis']
                
                if 'vehicle_count' in analysis:
                    vehicle_counts.append(analysis['vehicle_count'])
                
                if 'pedestrian_count' in analysis:
                    pedestrian_counts.append(analysis['pedestrian_count'])
                
                if 'congestion_percent' in analysis:
                    congestion_levels.append(analysis['congestion_percent'] / 100.0)
        
        self.vehicle_count = int(np.mean(vehicle_counts)) if vehicle_counts else 0
        self.pedestrian_count = int(np.mean(pedestrian_counts)) if pedestrian_counts else 0
        self.avg_congestion = np.mean(congestion_levels) if congestion_levels else 0
        self.max_congestion = np.max(congestion_levels) if congestion_levels else 0
        self.congestion_levels = congestion_levels
    
    def aggregate_insights(self):
        """Aggregate all insights from AI analyses"""
        all_safety_issues = []
        all_infrastructure_issues = []
        all_recommendations = []
        all_immediate_actions = []
        vehicle_types_total = defaultdict(int)
        risk_levels = []
        traffic_flows = []
        
        for result in self.frame_analyses:
            if not result or not result['analysis']:
                continue
            
            analysis = result['analysis']
            
            # Collect safety issues
            if 'safety_issues' in analysis and analysis['safety_issues']:
                all_safety_issues.extend(analysis['safety_issues'])
            
            # Collect infrastructure issues
            if 'infrastructure_issues' in analysis and analysis['infrastructure_issues']:
                all_infrastructure_issues.extend(analysis['infrastructure_issues'])
            
            # Collect recommendations
            if 'recommendations' in analysis and analysis['recommendations']:
                all_recommendations.extend(analysis['recommendations'])
            
            # Collect immediate actions
            if 'immediate_actions' in analysis and analysis['immediate_actions']:
                all_immediate_actions.extend(analysis['immediate_actions'])
            
            # Aggregate vehicle types
            if 'vehicle_types' in analysis and isinstance(analysis['vehicle_types'], dict):
                for vtype, count in analysis['vehicle_types'].items():
                    vehicle_types_total[vtype] += count
            
            # Collect risk levels
            if 'risk_level' in analysis:
                risk_levels.append(analysis['risk_level'])
            
            # Collect traffic flows
            if 'traffic_flow' in analysis:
                traffic_flows.append(analysis['traffic_flow'])
        
        return {
            'safety_issues': list(set(all_safety_issues)),
            'infrastructure_issues': list(set(all_infrastructure_issues)),
            'recommendations': list(set(all_recommendations)),
            'immediate_actions': list(set(all_immediate_actions)),
            'vehicle_types': dict(vehicle_types_total),
            'risk_levels': risk_levels,
            'traffic_flows': traffic_flows,
            'frames_analyzed': len(self.frame_analyses)
        }
    
    def generate_recommendations(self):
        """Generate comprehensive recommendations based on AI analysis"""
        insights = self.aggregate_insights()
        issues = []
        recommendations = []
        
        # Priority 1: Immediate safety actions from AI
        if insights['immediate_actions']:
            for action in insights['immediate_actions'][:5]:
                issues.append({
                    'type': 'IMMEDIATE SAFETY CONCERN',
                    'severity': 'CRITICAL',
                    'description': action
                })
                recommendations.append({
                    'category': 'Immediate Action Required',
                    'priority': 'CRITICAL',
                    'suggestion': action,
                    'impact': 'Address immediate safety hazard',
                    'cost': 'Varies - Immediate attention required'
                })
        
        # Priority 2: Safety issues from AI
        if insights['safety_issues']:
            for issue in insights['safety_issues'][:5]:
                issues.append({
                    'type': 'Safety Issue',
                    'severity': 'HIGH',
                    'description': issue
                })
        
        # Priority 3: Infrastructure issues from AI
        if insights['infrastructure_issues']:
            for issue in insights['infrastructure_issues'][:5]:
                issues.append({
                    'type': 'Infrastructure Issue',
                    'severity': 'MEDIUM',
                    'description': issue
                })
        
        # Priority 4: AI recommendations
        if insights['recommendations']:
            for rec in insights['recommendations'][:7]:
                recommendations.append({
                    'category': 'AI-Recommended Improvement',
                    'priority': 'HIGH',
                    'suggestion': rec,
                    'impact': 'Based on detailed visual traffic analysis',
                    'cost': 'See detailed cost estimate'
                })
        
        # Add congestion-based recommendations
        if self.avg_congestion > 0.7:
            issues.append({
                'type': 'High Congestion',
                'severity': 'HIGH',
                'description': f'Average congestion at {self.avg_congestion*100:.1f}% - Critical levels'
            })
            recommendations.append({
                'category': 'Traffic Flow Management',
                'priority': 'HIGH',
                'suggestion': 'Deploy adaptive traffic light system with AI-based timing optimization',
                'impact': 'Reduce congestion by 25-35%',
                'cost': 'Medium-High ($75k-$150k)'
            })
        
        # Risk assessment from AI
        if insights['risk_levels']:
            high_risk_count = sum(1 for r in insights['risk_levels'] if r in ['high', 'critical'])
            if high_risk_count > len(insights['risk_levels']) * 0.3:
                recommendations.append({
                    'category': 'Safety Enhancement',
                    'priority': 'CRITICAL',
                    'suggestion': 'Implement comprehensive safety audit and immediate interventions',
                    'impact': 'Significantly reduce accident risk',
                    'cost': 'High ($50k-$200k depending on scope)'
                })
        
        return issues, recommendations
    
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
        insights = self.aggregate_insights()
        
        if not issues:
            summary = "AI analysis indicates traffic conditions are generally acceptable. "
        else:
            summary = f"AI-powered analysis identified {len(issues)} significant issue(s). "
        
        summary += f"Analyzed {insights['frames_analyzed']} frames across {self.duration:.1f} seconds of footage. "
        summary += f"Traffic congestion level: {self.get_congestion_rating()}. "
        
        # Add traffic flow summary
        if insights['traffic_flows']:
            flow_summary = defaultdict(int)
            for flow in insights['traffic_flows']:
                flow_summary[flow] += 1
            dominant_flow = max(flow_summary.items(), key=lambda x: x[1])[0]
            summary += f"Dominant traffic flow state: {dominant_flow}. "
        
        summary += f"Generated {len(recommendations)} evidence-based recommendation(s)."
        
        return summary
    
    def generate_report(self):
        """Generate comprehensive analysis report"""
        issues, recommendations = self.generate_recommendations()
        insights = self.aggregate_insights()
        
        report = {
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'video_info': {
                'path': self.video_path,
                'duration_seconds': round(self.duration, 2),
                'fps': self.fps,
                'total_frames': self.total_frames,
                'frames_analyzed': insights['frames_analyzed']
            },
            'metrics': {
                'avg_vehicles_per_frame': self.vehicle_count,
                'vehicle_type_distribution': insights['vehicle_types'],
                'avg_pedestrians_per_frame': self.pedestrian_count,
                'avg_congestion_level': round(self.avg_congestion, 3),
                'max_congestion_level': round(self.max_congestion, 3),
                'congestion_rating': self.get_congestion_rating()
            },
            'ai_insights': {
                'safety_issues_detected': insights['safety_issues'],
                'infrastructure_issues': insights['infrastructure_issues'],
                'risk_level_distribution': dict(defaultdict(int, [(r, insights['risk_levels'].count(r)) for r in set(insights['risk_levels'])])),
                'traffic_flow_states': insights['traffic_flows']
            },
            'issues_detected': issues,
            'recommendations': recommendations,
            'summary': self.generate_summary(issues, recommendations),
            'detailed_frame_analyses': self.frame_analyses
        }
        
        return report
    
    def print_report(self):
        """Print formatted report to console"""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("AI-POWERED SMART TRAFFIC ANALYSIS REPORT".center(80))
        print("=" * 80)
        
        print(f"\n📅 Analysis Date: {report['analysis_date']}")
        print(f"🎥 Video: {report['video_info']['path']}")
        print(f"⏱️  Duration: {report['video_info']['duration_seconds']}s")
        print(f"🤖 AI Frames Analyzed: {report['video_info']['frames_analyzed']}")
        
        print("\n" + "-" * 80)
        print("TRAFFIC METRICS (AI-DETECTED)".center(80))
        print("-" * 80)
        
        metrics = report['metrics']
        print(f"🚗 Average Vehicles per Frame: {metrics['avg_vehicles_per_frame']}")
        
        if metrics['vehicle_type_distribution']:
            print(f"🚙 Vehicle Types Detected:")
            for vtype, count in metrics['vehicle_type_distribution'].items():
                print(f"   - {vtype.capitalize()}: {count}")
        
        print(f"🚶 Average Pedestrians per Frame: {metrics['avg_pedestrians_per_frame']}")
        print(f"📊 Average Congestion Level: {metrics['avg_congestion_level']*100:.1f}%")
        print(f"📈 Peak Congestion Level: {metrics['max_congestion_level']*100:.1f}%")
        print(f"⚠️  Congestion Rating: {metrics['congestion_rating']}")
        
        # AI Insights
        print("\n" + "-" * 80)
        print("AI INSIGHTS & OBSERVATIONS".center(80))
        print("-" * 80)
        
        ai_insights = report['ai_insights']
        
        if ai_insights['safety_issues_detected']:
            print(f"\n🚨 Safety Issues Detected ({len(ai_insights['safety_issues_detected'])}):")
            for i, issue in enumerate(ai_insights['safety_issues_detected'][:10], 1):
                print(f"   {i}. {issue}")
        
        if ai_insights['infrastructure_issues']:
            print(f"\n🏗️  Infrastructure Issues ({len(ai_insights['infrastructure_issues'])}):")
            for i, issue in enumerate(ai_insights['infrastructure_issues'][:10], 1):
                print(f"   {i}. {issue}")
        
        if ai_insights['risk_level_distribution']:
            print(f"\n⚠️  Risk Level Distribution:")
            for level, count in ai_insights['risk_level_distribution'].items():
                print(f"   - {level.upper()}: {count} frames")
        
        # Issues
        if report['issues_detected']:
            print("\n" + "-" * 80)
            print("ISSUES DETECTED".center(80))
            print("-" * 80)
            
            for i, issue in enumerate(report['issues_detected'], 1):
                print(f"\n{i}. [{issue['severity']}] {issue['type']}")
                print(f"   {issue['description']}")
        
        # Recommendations
        if report['recommendations']:
            print("\n" + "-" * 80)
            print("RECOMMENDATIONS".center(80))
            print("-" * 80)
            
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"\n{i}. [{rec['priority']}] {rec['category']}")
                print(f"   💡 {rec['suggestion']}")
                print(f"   📊 Impact: {rec['impact']}")
                print(f"   💰 Cost: {rec['cost']}")
        
        # Summary
        print("\n" + "-" * 80)
        print("EXECUTIVE SUMMARY".center(80))
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
        print(f"\n📄 Detailed report saved to: {output_path}")


def main():
    """Main function"""
    print("=" * 80)
    print("AI-POWERED SMART TRAFFIC ANALYZER".center(80))
    print("=" * 80)
    
    # Get OpenAI API key
    api_key = input("\n🔑 Enter your OpenAI API key (or press Enter to use OPENAI_API_KEY env var): ").strip()
    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("❌ Error: OpenAI API key is required!")
            print("Either provide it when prompted or set the OPENAI_API_KEY environment variable.")
            return
    
    # Get video path
    video_path = input("\n🎥 Enter the path to your MP4 traffic video: ").strip()
    
    # Get sample interval
    print("\n⚙️  Sample Interval Configuration:")
    print("   - Lower interval = More frames analyzed = More accurate but slower and more expensive")
    print("   - Higher interval = Fewer frames analyzed = Faster and cheaper but less detailed")
    print("   - Recommended: 30 (analyzes ~1 frame per second for 30fps video)")
    
    sample_input = input("   Enter sample interval (press Enter for default 30): ").strip()
    sample_interval = int(sample_input) if sample_input.isdigit() else 30
    
    # Initialize analyzer
    try:
        print(f"\n🔧 Initializing analyzer with OpenAI API...")
        analyzer = TrafficAnalyzer(video_path, openai_api_key=api_key)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        return
    
    # Analyze video
    print("\n🔍 Starting AI-powered analysis...")
    print("⏳ This may take several minutes depending on video length...")
    print()
    
    try:
        analyzer.analyze_video(sample_interval=sample_interval)
    except Exception as e:
        print(f"\n❌ Analysis error: {str(e)}")
        return
    
    # Print report
    analyzer.print_report()
    
    # Save report
    save_option = input("\n💾 Save detailed report to JSON file? (y/n): ").strip().lower()
    if save_option == 'y':
        filename = input("Enter filename (default: traffic_analysis_report.json): ").strip()
        if not filename:
            filename = 'traffic_analysis_report.json'
        analyzer.save_report(filename)
    
    print("\n✅ Analysis complete!")
    print(f"📊 Total OpenAI API calls made: {len(analyzer.frame_analyses)}")


if __name__ == "__main__":
    main()
