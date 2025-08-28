"""
Performance Metrics and Evaluation Tools
========================================

This module provides comprehensive metrics for evaluating draft strategies
and seasonal performance prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

from ..core.player import Player


@dataclass
class WeeklyPerformance:
    """Weekly fantasy performance data"""
    week: int
    points_scored: float
    projected_points: float
    actual_points: Optional[float] = None
    injuries: List[str] = None
    bye_week_players: List[str] = None


@dataclass
class SeasonMetrics:
    """Comprehensive season performance metrics"""
    total_points: float
    avg_points_per_week: float
    consistency_score: float  # Lower std dev is better
    injury_adjusted_points: float
    bye_week_penalty: float
    position_balance_score: float
    draft_efficiency: float
    final_rank: Optional[int] = None
    playoff_made: Optional[bool] = None
    championship_won: Optional[bool] = None


class DraftMetrics:
    """
    Calculate comprehensive draft performance metrics
    """
    
    @staticmethod
    def calculate_roster_balance(roster: List[Player], 
                                league_settings: Dict) -> float:
        """
        Calculate how well-balanced a roster is across positions.
        
        Returns score from 0.0 (terrible) to 1.0 (perfect balance)
        """
        
        position_counts = defaultdict(int)
        for player in roster:
            position_counts[player.position] += 1
        
        # Ideal roster composition for 12-team league
        ideal_composition = league_settings.get('roster_spots', {
            'QB': 2, 'RB': 4, 'WR': 5, 'TE': 2, 'K': 1, 'DEF': 1
        })
        
        balance_scores = []
        
        for position, ideal_count in ideal_composition.items():
            actual_count = position_counts.get(position, 0)
            
            # Calculate deviation from ideal
            if ideal_count > 0:
                deviation = abs(actual_count - ideal_count) / ideal_count
                position_score = max(0.0, 1.0 - deviation)
            else:
                position_score = 1.0 if actual_count == 0 else 0.0
            
            balance_scores.append(position_score)
        
        return np.mean(balance_scores)
    
    @staticmethod
    def calculate_draft_efficiency(roster: List[Player], 
                                  pick_order: List[int]) -> float:
        """
        Calculate how efficiently draft picks were used based on VORP vs ADP.
        
        Higher score = better value picks relative to draft position
        """
        
        if len(roster) != len(pick_order):
            return 0.0
        
        efficiency_scores = []
        
        for player, pick_number in zip(roster, pick_order):
            player_adp = getattr(player, 'adp_rank', pick_number)
            
            # Value = how much earlier the player was available vs when picked
            value_gained = max(0, player_adp - pick_number)
            
            # Normalize by typical ADP variance (±20 picks)
            normalized_value = min(1.0, value_gained / 20.0)
            efficiency_scores.append(normalized_value)
        
        return np.mean(efficiency_scores)
    
    @staticmethod
    def calculate_bye_week_optimization(roster: List[Player]) -> Dict[str, float]:
        """
        Analyze bye week distribution and calculate optimization metrics.
        
        Returns dict with various bye week metrics
        """
        
        bye_weeks = []
        position_bye_map = defaultdict(list)
        
        for player in roster:
            bye_week = getattr(player, 'bye_week', 0)
            if bye_week > 0:
                bye_weeks.append(bye_week)
                position_bye_map[player.position].append(bye_week)
        
        if not bye_weeks:
            return {'conflict_score': 1.0, 'concentration_score': 0.5, 'coverage_score': 1.0}
        
        # Conflict score (fewer conflicts = better)
        unique_weeks = len(set(bye_weeks))
        total_players = len(bye_weeks)
        conflict_score = unique_weeks / total_players if total_players > 0 else 1.0
        
        # Concentration analysis
        bye_week_counts = defaultdict(int)
        for week in bye_weeks:
            bye_week_counts[week] += 1
        
        max_concentration = max(bye_week_counts.values())
        
        # Good concentration: 3-4 players in one week, spread elsewhere
        if max_concentration in [3, 4] and len(bye_week_counts) <= 4:
            concentration_score = 1.0
        elif max_concentration <= 2:
            concentration_score = 0.8  # Good spread
        else:
            concentration_score = max(0.0, 1.0 - (max_concentration - 4) * 0.2)
        
        # Position coverage (don't lose all RBs/WRs same week)
        coverage_penalties = 0
        for position, weeks in position_bye_map.items():
            if position in ['RB', 'WR'] and len(set(weeks)) == 1 and len(weeks) >= 3:
                coverage_penalties += 1
        
        coverage_score = max(0.0, 1.0 - coverage_penalties * 0.3)
        
        return {
            'conflict_score': conflict_score,
            'concentration_score': concentration_score,
            'coverage_score': coverage_score,
            'overall_bye_score': np.mean([conflict_score, concentration_score, coverage_score])
        }
    
    @staticmethod
    def calculate_injury_risk_profile(roster: List[Player]) -> Dict[str, float]:
        """
        Calculate injury risk metrics for the roster.
        """
        
        injury_risks = []
        durability_scores = []
        high_risk_count = 0
        
        for player in roster:
            injury_risk = player.metadata.get('injury_risk_score', 0.3)
            durability = player.metadata.get('durability_score', 0.7)
            
            injury_risks.append(injury_risk)
            durability_scores.append(durability)
            
            if injury_risk > 0.6:  # High risk threshold
                high_risk_count += 1
        
        avg_injury_risk = np.mean(injury_risks) if injury_risks else 0.3
        avg_durability = np.mean(durability_scores) if durability_scores else 0.7
        
        # Risk diversity (having all high-risk players is bad)
        risk_std = np.std(injury_risks) if len(injury_risks) > 1 else 0.1
        diversity_score = min(1.0, risk_std * 3)  # Higher std = better diversity
        
        # Overall risk score (lower injury risk + higher durability = better)
        overall_risk_score = (1.0 - avg_injury_risk) * 0.6 + avg_durability * 0.4
        
        return {
            'avg_injury_risk': avg_injury_risk,
            'avg_durability': avg_durability,
            'high_risk_count': high_risk_count,
            'risk_diversity': diversity_score,
            'overall_risk_score': overall_risk_score
        }
    
    @staticmethod
    def calculate_upside_potential(roster: List[Player]) -> Dict[str, float]:
        """
        Calculate potential upside metrics (rookies, breakout candidates).
        """
        
        rookie_count = 0
        rookie_upside = 0.0
        total_uncertainty = 0.0
        
        for player in roster:
            is_rookie = player.metadata.get('is_rookie', False)
            risk_sigma = player.metadata.get('risk_sigma', 0.2)
            
            if is_rookie:
                rookie_count += 1
                rookie_upside += player.vorp * risk_sigma  # Higher uncertainty = more upside
            
            total_uncertainty += risk_sigma
        
        avg_uncertainty = total_uncertainty / len(roster) if roster else 0.0
        
        # Upside score balances rookie potential vs risk
        upside_score = min(1.0, rookie_upside / 10.0) if rookie_upside > 0 else 0.0
        
        return {
            'rookie_count': rookie_count,
            'rookie_upside': rookie_upside,
            'avg_uncertainty': avg_uncertainty,
            'upside_score': upside_score
        }


class SeasonPerformanceEvaluator:
    """
    Evaluate draft strategy performance over a full fantasy season.
    """
    
    def __init__(self, league_settings: Dict):
        self.league_settings = league_settings
        self.weekly_performances = []
    
    def simulate_season(self, 
                       roster: List[Player],
                       n_weeks: int = 17,
                       injury_rate: float = 0.15) -> SeasonMetrics:
        """
        Simulate a full fantasy season for a drafted roster.
        
        Args:
            roster: Drafted players
            n_weeks: Number of weeks to simulate
            injury_rate: Probability of injury per player per week
            
        Returns:
            Comprehensive season metrics
        """
        
        weekly_scores = []
        total_injury_weeks = 0
        total_bye_week_penalties = 0
        
        # Create initial active roster
        active_roster = self._create_starting_lineup(roster)
        
        for week in range(1, n_weeks + 1):
            # Simulate week
            week_performance = self._simulate_week(
                active_roster, roster, week, injury_rate
            )
            
            weekly_scores.append(week_performance.points_scored)
            
            if week_performance.injuries:
                total_injury_weeks += len(week_performance.injuries)
            
            if week_performance.bye_week_players:
                total_bye_week_penalties += len(week_performance.bye_week_players)
            
            self.weekly_performances.append(week_performance)
        
        # Calculate season metrics
        total_points = sum(weekly_scores)
        avg_points = np.mean(weekly_scores)
        consistency = 1.0 / (1.0 + np.std(weekly_scores))  # Higher = more consistent
        
        # Injury adjustment
        injury_penalty = total_injury_weeks * 5.0  # 5 points per injury week
        injury_adjusted_points = total_points - injury_penalty
        
        # Bye week penalty
        bye_week_penalty = total_bye_week_penalties * 2.0  # 2 points per bye week issue
        
        # Position balance (affects consistency)
        balance_score = DraftMetrics.calculate_roster_balance(roster, self.league_settings)
        
        # Draft efficiency
        pick_order = list(range(1, len(roster) + 1))  # Simplified
        draft_efficiency = DraftMetrics.calculate_draft_efficiency(roster, pick_order)
        
        return SeasonMetrics(
            total_points=total_points,
            avg_points_per_week=avg_points,
            consistency_score=consistency,
            injury_adjusted_points=injury_adjusted_points,
            bye_week_penalty=bye_week_penalty,
            position_balance_score=balance_score,
            draft_efficiency=draft_efficiency
        )
    
    def _create_starting_lineup(self, roster: List[Player]) -> List[Player]:
        """Create optimal starting lineup from roster"""
        
        # Simple: take highest VORP players by position
        lineup = []
        position_needs = {'QB': 1, 'RB': 2, 'WR': 2, 'TE': 1, 'K': 1, 'DEF': 1}
        
        # Sort roster by VORP within each position
        position_players = defaultdict(list)
        for player in roster:
            position_players[player.position].append(player)
        
        for position, players in position_players.items():
            players.sort(key=lambda p: p.vorp, reverse=True)
        
        # Fill starting lineup
        for position, needed in position_needs.items():
            available = position_players.get(position, [])
            for i in range(min(needed, len(available))):
                lineup.append(available[i])
        
        # Add FLEX (best remaining RB/WR/TE)
        flex_candidates = []
        for position in ['RB', 'WR', 'TE']:
            available = position_players.get(position, [])
            used_count = position_needs.get(position, 0)
            for i in range(used_count, len(available)):
                flex_candidates.append(available[i])
        
        if flex_candidates:
            best_flex = max(flex_candidates, key=lambda p: p.vorp)
            lineup.append(best_flex)
        
        return lineup
    
    def _simulate_week(self, 
                      active_lineup: List[Player],
                      full_roster: List[Player],
                      week: int,
                      injury_rate: float) -> WeeklyPerformance:
        """Simulate performance for a single week"""
        
        week_points = 0.0
        injuries = []
        bye_week_players = []
        
        for player in active_lineup:
            # Check for bye week
            player_bye = getattr(player, 'bye_week', 0)
            if player_bye == week:
                bye_week_players.append(player.name)
                continue  # No points this week
            
            # Check for injury
            injury_risk = player.metadata.get('injury_risk_score', 0.15)
            if np.random.random() < injury_risk * injury_rate:
                injuries.append(player.name)
                week_points += player.proj_ppg * 0.3  # Partial points before injury
                continue
            
            # Normal performance with variance
            base_points = player.proj_ppg
            variance = player.metadata.get('risk_sigma', 0.2)
            actual_points = max(0, np.random.normal(base_points, base_points * variance))
            
            week_points += actual_points
        
        # Replace bye week/injured players from bench (simplified)
        replacements_needed = len(bye_week_players) + len(injuries)
        if replacements_needed > 0:
            bench_players = [p for p in full_roster if p not in active_lineup]
            available_replacements = [
                p for p in bench_players 
                if getattr(p, 'bye_week', 0) != week and p.name not in injuries
            ]
            
            # Add replacement points (typically lower quality)
            for i in range(min(replacements_needed, len(available_replacements))):
                replacement = available_replacements[i]
                replacement_points = replacement.proj_ppg * 0.7  # Bench players typically score less
                week_points += replacement_points
        
        return WeeklyPerformance(
            week=week,
            points_scored=week_points,
            projected_points=sum(p.proj_ppg for p in active_lineup),
            injuries=injuries,
            bye_week_players=bye_week_players
        )
    
    def create_season_report(self, 
                           season_metrics: SeasonMetrics,
                           roster: List[Player]) -> str:
        """Create detailed season performance report"""
        
        report_lines = [
            "Fantasy Football Season Performance Report",
            "=" * 50,
            "",
            f"📊 SEASON SUMMARY",
            "-" * 20,
            f"Total Points: {season_metrics.total_points:.1f}",
            f"Average Per Week: {season_metrics.avg_points_per_week:.1f}",
            f"Consistency Score: {season_metrics.consistency_score:.3f}",
            f"Injury-Adjusted Points: {season_metrics.injury_adjusted_points:.1f}",
            f"Bye Week Penalty: {season_metrics.bye_week_penalty:.1f}",
            "",
            f"🏆 ROSTER ANALYSIS",
            "-" * 20,
            f"Position Balance: {season_metrics.position_balance_score:.3f}",
            f"Draft Efficiency: {season_metrics.draft_efficiency:.3f}",
            ""
        ]
        
        # Roster breakdown
        position_counts = defaultdict(int)
        total_vorp = 0.0
        
        for player in roster:
            position_counts[player.position] += 1
            total_vorp += player.vorp
        
        report_lines.extend([
            f"📋 ROSTER COMPOSITION",
            "-" * 25,
            f"Total VORP: {total_vorp:.2f}"
        ])
        
        for position, count in sorted(position_counts.items()):
            position_players = [p for p in roster if p.position == position]
            avg_vorp = np.mean([p.vorp for p in position_players])
            report_lines.append(f"{position}: {count} players (Avg VORP: {avg_vorp:.2f})")
        
        # Weekly performance summary
        if self.weekly_performances:
            report_lines.extend([
                "",
                f"📈 WEEKLY PERFORMANCE HIGHLIGHTS",
                "-" * 35
            ])
            
            scores = [wp.points_scored for wp in self.weekly_performances]
            best_week = max(scores)
            worst_week = min(scores)
            
            total_injuries = sum(len(wp.injuries) for wp in self.weekly_performances if wp.injuries)
            total_bye_issues = sum(len(wp.bye_week_players) for wp in self.weekly_performances if wp.bye_week_players)
            
            report_lines.extend([
                f"Best Week: {best_week:.1f} points",
                f"Worst Week: {worst_week:.1f} points",
                f"Total Injury Weeks: {total_injuries}",
                f"Total Bye Week Issues: {total_bye_issues}"
            ])
        
        return "\n".join(report_lines)
    
    def visualize_season_performance(self, 
                                   season_metrics: SeasonMetrics,
                                   save_path: Optional[str] = None):
        """Create visualizations of season performance"""
        
        if not self.weekly_performances:
            print("No weekly performance data available for visualization")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Fantasy Football Season Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Weekly scores
        ax1 = axes[0, 0]
        weeks = [wp.week for wp in self.weekly_performances]
        scores = [wp.points_scored for wp in self.weekly_performances]
        projected = [wp.projected_points for wp in self.weekly_performances]
        
        ax1.plot(weeks, scores, 'b-o', label='Actual', linewidth=2)
        ax1.plot(weeks, projected, 'r--', label='Projected', alpha=0.7)
        ax1.set_title('Weekly Performance')
        ax1.set_xlabel('Week')
        ax1.set_ylabel('Fantasy Points')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Score distribution
        ax2 = axes[0, 1]
        ax2.hist(scores, bins=8, alpha=0.7, color='skyblue', edgecolor='navy')
        ax2.axvline(np.mean(scores), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(scores):.1f}')
        ax2.set_title('Score Distribution')
        ax2.set_xlabel('Fantasy Points')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Injury/Bye week impact
        ax3 = axes[1, 0]
        injury_weeks = [len(wp.injuries) if wp.injuries else 0 for wp in self.weekly_performances]
        bye_weeks = [len(wp.bye_week_players) if wp.bye_week_players else 0 for wp in self.weekly_performances]
        
        ax3.bar([w - 0.2 for w in weeks], injury_weeks, width=0.4, 
               label='Injuries', color='red', alpha=0.7)
        ax3.bar([w + 0.2 for w in weeks], bye_weeks, width=0.4, 
               label='Bye Weeks', color='orange', alpha=0.7)
        ax3.set_title('Roster Disruptions by Week')
        ax3.set_xlabel('Week')
        ax3.set_ylabel('Number of Players Affected')
        ax3.legend()
        
        # 4. Performance metrics
        ax4 = axes[1, 1]
        metrics = {
            'Total Points': season_metrics.total_points,
            'Consistency': season_metrics.consistency_score * 100,  # Scale for visibility
            'Balance Score': season_metrics.position_balance_score * 100,
            'Draft Efficiency': season_metrics.draft_efficiency * 100,
            'Injury-Adj Points': season_metrics.injury_adjusted_points
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = ax4.bar(range(len(metrics)), metric_values, 
                      color=['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum'])
        ax4.set_title('Season Metrics Summary')
        ax4.set_xticks(range(len(metrics)))
        ax4.set_xticklabels(metric_names, rotation=45, ha='right')
        ax4.set_ylabel('Score/Points')
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'{value:.1f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Season performance visualization saved: {save_path}")
        
        plt.show()


if __name__ == "__main__":
    # Example usage
    from ..utils.data_loader import create_sample_player_pool, get_default_league_settings
    
    print("📊 Testing Metrics and Season Evaluation")
    
    # Create sample roster
    player_pool = create_sample_player_pool(200)
    sample_roster = list(player_pool)[:15]  # Take first 15 as drafted roster
    
    league_settings = get_default_league_settings()
    
    # Test draft metrics
    print("\n📋 Draft Metrics:")
    balance_score = DraftMetrics.calculate_roster_balance(sample_roster, league_settings.roster_spots)
    print(f"Roster Balance: {balance_score:.3f}")
    
    efficiency_score = DraftMetrics.calculate_draft_efficiency(sample_roster, list(range(1, 16)))
    print(f"Draft Efficiency: {efficiency_score:.3f}")
    
    bye_metrics = DraftMetrics.calculate_bye_week_optimization(sample_roster)
    print(f"Bye Week Score: {bye_metrics['overall_bye_score']:.3f}")
    
    injury_metrics = DraftMetrics.calculate_injury_risk_profile(sample_roster)
    print(f"Injury Risk Score: {injury_metrics['overall_risk_score']:.3f}")
    
    # Test season simulation
    print("\n🏈 Season Simulation:")
    evaluator = SeasonPerformanceEvaluator(league_settings.roster_spots)
    season_metrics = evaluator.simulate_season(sample_roster)
    
    print(f"Total Points: {season_metrics.total_points:.1f}")
    print(f"Avg Per Week: {season_metrics.avg_points_per_week:.1f}")
    print(f"Consistency: {season_metrics.consistency_score:.3f}")
    
    # Create reports
    season_report = evaluator.create_season_report(season_metrics, sample_roster)
    print(f"\n📋 Season Report Preview:")
    print(season_report[:500] + "...")
    
    # Visualize
    evaluator.visualize_season_performance(season_metrics, "season_performance_demo.png")
    
    print("\n✅ Metrics and evaluation demo complete!")
