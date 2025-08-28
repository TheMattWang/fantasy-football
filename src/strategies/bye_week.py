"""
Advanced Bye Week Management for Fantasy Football Drafts
========================================================

This module provides sophisticated bye week optimization that goes far beyond
simple conflict avoidance to create strategic bye week advantages.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict, Counter
from dataclasses import dataclass
from itertools import combinations

from ..core.player import Player, PlayerPool
from ..core.draft import DraftState, LeagueSettings


@dataclass
class ByeWeekAnalysis:
    """Analysis of bye week distribution and conflicts"""
    week: int
    player_count: int  # Total players with this bye week
    position_distribution: Dict[str, int]  # Position -> count
    tier_distribution: Dict[str, int]  # Tier -> count (e.g., "elite", "solid", "depth")
    avg_vorp: float
    conflict_risk: float  # How likely conflicts are for this week
    strategic_value: float  # Strategic value of concentrating or avoiding this week


class ByeWeekOptimizer:
    """
    Advanced bye week management system that considers:
    
    1. **Strategic Concentration**: Sometimes clustering bye weeks is advantageous
    2. **Tier Balancing**: Ensure you don't lose all top players same week
    3. **Positional Coverage**: Maintain coverage across all positions
    4. **Waiver Wire Opportunities**: Consider available replacements
    5. **Opponent Analysis**: Account for league-wide bye week conflicts
    """
    
    def __init__(self, league_settings: LeagueSettings):
        self.league_settings = league_settings
        self.bye_week_analysis: Dict[int, ByeWeekAnalysis] = {}
        self.optimal_strategies: Dict[str, float] = {}  # Strategy -> score
    
    def analyze_bye_week_landscape(self, player_pool: PlayerPool) -> None:
        """Analyze the entire bye week landscape for strategic planning"""
        
        print("📅 Analyzing bye week landscape...")
        
        # Group players by bye week
        bye_week_groups = defaultdict(list)
        for player in player_pool:
            bye_week = getattr(player, 'bye_week', 0)
            if bye_week > 0:
                bye_week_groups[bye_week].append(player)
        
        # Analyze each bye week
        for week, players in bye_week_groups.items():
            self.bye_week_analysis[week] = self._analyze_single_bye_week(week, players)
        
        # Determine optimal strategies
        self._calculate_optimal_strategies()
        
        print(f"✅ Analyzed {len(self.bye_week_analysis)} bye weeks")
        self._print_bye_week_summary()
    
    def _analyze_single_bye_week(self, week: int, players: List[Player]) -> ByeWeekAnalysis:
        """Analyze a single bye week in detail"""
        
        # Position distribution
        position_dist = Counter(p.position for p in players)
        
        # Tier distribution (based on VORP)
        sorted_players = sorted(players, key=lambda p: p.vorp, reverse=True)
        n_players = len(sorted_players)
        tier_dist = {
            'elite': 0,     # Top 20%
            'solid': 0,     # Next 40%
            'depth': 0,     # Bottom 40%
        }
        
        for i, player in enumerate(sorted_players):
            if i < n_players * 0.2:
                tier_dist['elite'] += 1
            elif i < n_players * 0.6:
                tier_dist['solid'] += 1
            else:
                tier_dist['depth'] += 1
        
        # Calculate metrics
        avg_vorp = np.mean([p.vorp for p in players])
        
        # Conflict risk: higher when many players, especially elite ones
        conflict_risk = min(1.0, (tier_dist['elite'] * 3 + tier_dist['solid']) / 20.0)
        
        # Strategic value: balance of talent concentration vs. spread
        strategic_value = self._calculate_strategic_value(tier_dist, position_dist)
        
        return ByeWeekAnalysis(
            week=week,
            player_count=len(players),
            position_distribution=dict(position_dist),
            tier_distribution=tier_dist,
            avg_vorp=avg_vorp,
            conflict_risk=conflict_risk,
            strategic_value=strategic_value
        )
    
    def _calculate_strategic_value(self, tier_dist: Dict[str, int], position_dist: Dict[str, int]) -> float:
        """Calculate strategic value of a bye week"""
        
        # Higher value for weeks with good depth options
        depth_value = tier_dist['depth'] * 0.1
        
        # Moderate value for solid players
        solid_value = tier_dist['solid'] * 0.3
        
        # Elite players create both opportunity and risk
        elite_value = tier_dist['elite'] * 0.5
        
        # Position diversity bonus
        position_diversity = len(position_dist) * 0.2
        
        # Balance formula: reward depth while managing elite concentration
        strategic_value = depth_value + solid_value + (elite_value * 0.7) + position_diversity
        
        return strategic_value
    
    def _calculate_optimal_strategies(self) -> None:
        """Calculate optimal bye week strategies"""
        
        # Strategy 1: Concentration (cluster bye weeks)
        concentration_score = self._evaluate_concentration_strategy()
        
        # Strategy 2: Spread (avoid conflicts)
        spread_score = self._evaluate_spread_strategy()
        
        # Strategy 3: Strategic Clustering (cluster in optimal weeks)
        strategic_score = self._evaluate_strategic_clustering()
        
        self.optimal_strategies = {
            'concentration': concentration_score,
            'spread': spread_score,
            'strategic_clustering': strategic_score
        }
    
    def _evaluate_concentration_strategy(self) -> float:
        """Evaluate clustering all/most players in 1-2 bye weeks"""
        
        # Find weeks with best strategic value
        sorted_weeks = sorted(
            self.bye_week_analysis.items(),
            key=lambda x: x[1].strategic_value,
            reverse=True
        )
        
        if len(sorted_weeks) >= 2:
            # Score based on top 2 weeks
            top_weeks = sorted_weeks[:2]
            score = sum(analysis.strategic_value for _, analysis in top_weeks)
            
            # Bonus for having good depth in these weeks
            depth_bonus = sum(analysis.tier_distribution['depth'] for _, analysis in top_weeks) * 0.1
            
            return score + depth_bonus
        
        return 0.0
    
    def _evaluate_spread_strategy(self) -> float:
        """Evaluate spreading players across different bye weeks"""
        
        # Score based on availability of low-conflict weeks
        low_conflict_weeks = [
            analysis for analysis in self.bye_week_analysis.values()
            if analysis.conflict_risk < 0.3
        ]
        
        # More low-conflict weeks = higher spread strategy score
        spread_score = len(low_conflict_weeks) * 1.5
        
        # Bonus for even talent distribution
        if len(low_conflict_weeks) >= 4:
            spread_score += 2.0
        
        return spread_score
    
    def _evaluate_strategic_clustering(self) -> float:
        """Evaluate clustering in strategically optimal weeks"""
        
        # Find weeks that balance talent and depth
        balanced_weeks = [
            analysis for analysis in self.bye_week_analysis.values()
            if (analysis.tier_distribution['solid'] >= 3 and
                analysis.tier_distribution['depth'] >= 5 and
                analysis.conflict_risk < 0.5)
        ]
        
        if balanced_weeks:
            # Score highest balanced week
            best_week = max(balanced_weeks, key=lambda x: x.strategic_value)
            return best_week.strategic_value * 2.0
        
        return 0.0
    
    def get_optimal_strategy(self) -> str:
        """Get the optimal bye week strategy for current landscape"""
        if not self.optimal_strategies:
            return "spread"  # Default to spread
        
        return max(self.optimal_strategies.items(), key=lambda x: x[1])[0]
    
    def calculate_bye_week_penalty(self, player: Player, current_roster: List[Player]) -> float:
        """
        Calculate sophisticated bye week penalty/bonus for adding a player.
        
        Returns negative values for penalties, positive for bonuses.
        """
        
        player_bye = getattr(player, 'bye_week', 0)
        if player_bye == 0:
            return 0.0
        
        # Get current bye week distribution
        current_byes = [getattr(p, 'bye_week', 0) for p in current_roster if getattr(p, 'bye_week', 0) > 0]
        bye_counts = Counter(current_byes)
        
        optimal_strategy = self.get_optimal_strategy()
        
        if optimal_strategy == "concentration":
            return self._calculate_concentration_penalty(player, player_bye, bye_counts)
        elif optimal_strategy == "strategic_clustering":
            return self._calculate_strategic_clustering_penalty(player, player_bye, bye_counts)
        else:  # spread strategy
            return self._calculate_spread_penalty(player, player_bye, bye_counts)
    
    def _calculate_concentration_penalty(self, player: Player, player_bye: int, bye_counts: Counter) -> float:
        """Calculate penalty for concentration strategy"""
        
        # Find the target concentration week (highest count or best strategic value)
        if bye_counts:
            target_week = bye_counts.most_common(1)[0][0]
            target_count = bye_counts[target_week]
        else:
            # First player - choose best strategic week
            target_week = max(
                self.bye_week_analysis.items(),
                key=lambda x: x[1].strategic_value
            )[0]
            target_count = 0
        
        if player_bye == target_week:
            # Bonus for adding to concentration week
            return 0.3 + (target_count * 0.1)  # Increasing bonus
        else:
            # Penalty for spreading out
            return -0.2
    
    def _calculate_strategic_clustering_penalty(self, player: Player, player_bye: int, bye_counts: Counter) -> float:
        """Calculate penalty for strategic clustering"""
        
        player_analysis = self.bye_week_analysis.get(player_bye)
        if not player_analysis:
            return -0.1  # Small penalty for unknown bye week
        
        # Bonus for adding to strategically valuable weeks
        strategic_bonus = player_analysis.strategic_value * 0.1
        
        # Penalty for over-concentrating in any week
        current_count = bye_counts.get(player_bye, 0)
        if current_count >= 3:
            concentration_penalty = -(current_count - 2) * 0.2
        else:
            concentration_penalty = 0.0
        
        return strategic_bonus + concentration_penalty
    
    def _calculate_spread_penalty(self, player: Player, player_bye: int, bye_counts: Counter) -> float:
        """Calculate penalty for spread strategy"""
        
        current_count = bye_counts.get(player_bye, 0)
        
        if current_count == 0:
            # Bonus for new bye week
            return 0.2
        elif current_count == 1:
            # Small penalty for second player in same week
            return -0.1
        else:
            # Increasing penalty for concentration
            return -(current_count - 1) * 0.3
    
    def recommend_bye_week_targets(self, current_roster: List[Player], n_recommendations: int = 3) -> List[Tuple[int, str, float]]:
        """
        Recommend target bye weeks for future picks.
        
        Returns list of (bye_week, reason, score) tuples.
        """
        
        recommendations = []
        current_byes = Counter(getattr(p, 'bye_week', 0) for p in current_roster if getattr(p, 'bye_week', 0) > 0)
        optimal_strategy = self.get_optimal_strategy()
        
        for week, analysis in self.bye_week_analysis.items():
            current_count = current_byes.get(week, 0)
            
            if optimal_strategy == "concentration":
                # Favor week with highest current count or best strategic value
                if current_count > 0:
                    score = analysis.strategic_value + current_count * 0.5
                    reason = f"Continue concentration (have {current_count})"
                else:
                    score = analysis.strategic_value
                    reason = "High strategic value for concentration"
                    
            elif optimal_strategy == "strategic_clustering":
                # Favor balanced weeks without over-concentration
                if current_count >= 3:
                    score = 0.0  # Don't recommend over-concentrated weeks
                    reason = "Already concentrated"
                else:
                    score = analysis.strategic_value - (current_count * 0.2)
                    reason = f"Strategic balance (have {current_count})"
                    
            else:  # spread strategy
                # Favor weeks with no current players
                if current_count == 0:
                    score = analysis.strategic_value + 1.0
                    reason = "New bye week (spread strategy)"
                else:
                    score = analysis.strategic_value - (current_count * 0.5)
                    reason = f"Already have {current_count} players"
            
            recommendations.append((week, reason, score))
        
        # Sort by score and return top recommendations
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:n_recommendations]
    
    def _print_bye_week_summary(self) -> None:
        """Print summary of bye week analysis"""
        
        print(f"\n📅 Bye Week Landscape Summary:")
        print(f"=" * 35)
        
        # Show top strategic weeks
        sorted_weeks = sorted(
            self.bye_week_analysis.items(),
            key=lambda x: x[1].strategic_value,
            reverse=True
        )
        
        print(f"🔝 Top Strategic Bye Weeks:")
        for week, analysis in sorted_weeks[:5]:
            elite_count = analysis.tier_distribution['elite']
            solid_count = analysis.tier_distribution['solid']
            depth_count = analysis.tier_distribution['depth']
            print(f"   Week {week:2d}: {analysis.player_count:3d} players (E:{elite_count} S:{solid_count} D:{depth_count}) - Score: {analysis.strategic_value:.2f}")
        
        # Show optimal strategy
        optimal = self.get_optimal_strategy()
        optimal_score = self.optimal_strategies[optimal]
        print(f"\n🎯 Recommended Strategy: {optimal.replace('_', ' ').title()} (Score: {optimal_score:.2f})")
        
        if optimal == "concentration":
            print(f"   📍 Focus on clustering players in 1-2 high-value bye weeks")
        elif optimal == "strategic_clustering":
            print(f"   ⚖️  Balance talent across strategic bye weeks")
        else:
            print(f"   🌊 Spread players across different bye weeks to minimize conflicts")


class ByeWeekAwareMCTS:
    """
    MCTS enhanced with sophisticated bye week management.
    
    Integrates bye week optimization into MCTS decision making by:
    - Adjusting player valuations based on bye week strategy
    - Considering roster bye week balance in state evaluation
    - Planning multi-pick bye week strategies
    """
    
    def __init__(self, base_mcts, bye_week_optimizer: ByeWeekOptimizer, bye_week_weight: float = 0.2):
        """
        Initialize bye week aware MCTS.
        
        Args:
            base_mcts: Base MCTS implementation
            bye_week_optimizer: Trained bye week optimizer
            bye_week_weight: Weight for bye week considerations (0.0-1.0)
        """
        self.base_mcts = base_mcts
        self.bye_week_optimizer = bye_week_optimizer
        self.bye_week_weight = bye_week_weight
    
    def search(self, initial_state: DraftState) -> Optional[Player]:
        """Enhanced MCTS search with bye week optimization"""
        
        # Get base MCTS recommendation
        base_pick = self.base_mcts.search(initial_state)
        
        if not base_pick or self.bye_week_weight == 0:
            return base_pick
        
        # Enhance with bye week considerations
        available_players = list(initial_state.available_players)
        current_roster = initial_state.get_our_roster()
        
        enhanced_scores = []
        
        for player in available_players:
            base_score = self._get_base_score(player, initial_state)
            bye_week_adjustment = self.bye_week_optimizer.calculate_bye_week_penalty(player, current_roster)
            
            final_score = base_score + (bye_week_adjustment * self.bye_week_weight)
            enhanced_scores.append((player, final_score, bye_week_adjustment))
        
        # Sort by enhanced score
        enhanced_scores.sort(key=lambda x: x[1], reverse=True)
        bye_week_pick = enhanced_scores[0][0]
        bye_week_adjustment = enhanced_scores[0][2]
        
        # Log the decision if bye weeks influenced the choice
        if abs(bye_week_adjustment) > 0.1:
            player_bye = getattr(bye_week_pick, 'bye_week', 0)
            if bye_week_adjustment > 0:
                print(f"📅 Bye week bonus: {bye_week_pick.name} (Week {player_bye}) +{bye_week_adjustment:.2f}")
            else:
                print(f"📅 Bye week penalty: {bye_week_pick.name} (Week {player_bye}) {bye_week_adjustment:.2f}")
        
        return bye_week_pick
    
    def _get_base_score(self, player: Player, state: DraftState) -> float:
        """Get base score for a player (simplified)"""
        return player.vorp  # Simplified - would integrate with actual MCTS scoring
    
    def evaluate_roster_bye_week_strength(self, roster: List[Player]) -> Dict[str, float]:
        """Evaluate how well a roster handles bye weeks"""
        
        if not roster:
            return {'score': 0.0, 'strategy_alignment': 0.0, 'balance': 0.0}
        
        # Get bye week distribution
        bye_weeks = [getattr(p, 'bye_week', 0) for p in roster if getattr(p, 'bye_week', 0) > 0]
        bye_counts = Counter(bye_weeks)
        
        optimal_strategy = self.bye_week_optimizer.get_optimal_strategy()
        
        # Calculate strategy alignment
        if optimal_strategy == "concentration":
            # Reward clustering
            if bye_counts:
                max_concentration = max(bye_counts.values())
                strategy_alignment = min(1.0, max_concentration / 4.0)  # Cap at 4 players
            else:
                strategy_alignment = 0.0
        elif optimal_strategy == "spread":
            # Reward spreading
            unique_weeks = len(bye_counts)
            total_players = len(bye_weeks)
            if total_players > 0:
                strategy_alignment = min(1.0, unique_weeks / total_players)
            else:
                strategy_alignment = 1.0
        else:  # strategic_clustering
            # Reward moderate clustering in good weeks
            strategic_weeks = 0
            for week, count in bye_counts.items():
                if week in self.bye_week_optimizer.bye_week_analysis:
                    analysis = self.bye_week_optimizer.bye_week_analysis[week]
                    if analysis.strategic_value > 2.0 and 1 <= count <= 3:
                        strategic_weeks += 1
            strategy_alignment = min(1.0, strategic_weeks / 3.0)
        
        # Calculate balance (avoid having too many bye week conflicts)
        max_conflicts = max(bye_counts.values()) if bye_counts else 0
        balance = max(0.0, 1.0 - ((max_conflicts - 2) * 0.2))  # Penalty starts at 3+ conflicts
        
        # Overall score
        overall_score = (strategy_alignment * 0.6) + (balance * 0.4)
        
        return {
            'score': overall_score,
            'strategy_alignment': strategy_alignment,
            'balance': balance,
            'optimal_strategy': optimal_strategy,
            'bye_distribution': dict(bye_counts)
        }


def analyze_league_bye_week_landscape(player_pool: PlayerPool, league_settings: LeagueSettings) -> ByeWeekOptimizer:
    """
    Analyze and create bye week optimizer for a specific league.
    
    Args:
        player_pool: Available players
        league_settings: League configuration
        
    Returns:
        Configured ByeWeekOptimizer
    """
    
    optimizer = ByeWeekOptimizer(league_settings)
    optimizer.analyze_bye_week_landscape(player_pool)
    
    return optimizer


def create_bye_week_strategy_report(optimizer: ByeWeekOptimizer, current_roster: List[Player]) -> str:
    """Create a detailed bye week strategy report"""
    
    report = ["Bye Week Strategy Report", "=" * 30, ""]
    
    # Current roster analysis
    if current_roster:
        evaluation = ByeWeekAwareMCTS(None, optimizer, 0.0).evaluate_roster_bye_week_strength(current_roster)
        
        report.append("📊 Current Roster Bye Week Analysis:")
        report.append(f"   Overall Score: {evaluation['score']:.2f}/1.0")
        report.append(f"   Strategy Alignment: {evaluation['strategy_alignment']:.2f}/1.0")
        report.append(f"   Balance: {evaluation['balance']:.2f}/1.0")
        report.append(f"   Optimal Strategy: {evaluation['optimal_strategy'].replace('_', ' ').title()}")
        
        if evaluation['bye_distribution']:
            report.append(f"   Current Distribution: {evaluation['bye_distribution']}")
        report.append("")
    
    # Recommendations
    recommendations = optimizer.recommend_bye_week_targets(current_roster)
    report.append("🎯 Recommended Target Bye Weeks:")
    for week, reason, score in recommendations:
        report.append(f"   Week {week:2d}: {reason} (Score: {score:.2f})")
    report.append("")
    
    # Strategy explanation
    optimal_strategy = optimizer.get_optimal_strategy()
    report.append(f"📋 Strategy Guide ({optimal_strategy.replace('_', ' ').title()}):")
    
    if optimal_strategy == "concentration":
        report.extend([
            "   • Focus on clustering 4-6 players in 1-2 bye weeks",
            "   • Accept 'dead' weeks to maximize other weeks",
            "   • Target weeks with good waiver wire depth",
            "   • Easier to plan around and stream replacements"
        ])
    elif optimal_strategy == "strategic_clustering":
        report.extend([
            "   • Balance talent across 2-3 strategic bye weeks",
            "   • Avoid over-concentrating elite players",
            "   • Target weeks with good strategic value",
            "   • Maintain positional coverage each week"
        ])
    else:  # spread
        report.extend([
            "   • Spread players across different bye weeks",
            "   • Minimize conflicts and maintain weekly lineup strength",
            "   • More complex to manage but fewer 'dead' weeks",
            "   • Better for competitive leagues with limited waiver options"
        ])
    
    return "\n".join(report)


if __name__ == "__main__":
    # Example usage
    from ..core.draft import LeagueSettings
    from ..utils.data_loader import create_sample_player_pool
    
    league = LeagueSettings()
    player_pool = create_sample_player_pool(200)
    
    optimizer = analyze_league_bye_week_landscape(player_pool, league)
    print(f"\n{create_bye_week_strategy_report(optimizer, [])}")
