import numpy as np
import logging
from typing import Dict, Any, Tuple, List

logger = logging.getLogger("keter.negotiation_engine")

class CarlinOreCluster:
    def __init__(self, name: str, au: float, toc: float, tcm: float, arsenic: float, 
                 quartz: float, carbonates: float, bwi: float, fes2: float = 8.0):
        self.name = name
        self.au = au             # g/t (ppm)
        self.toc = toc           # wt%
        self.tcm = tcm           # wt%
        self.arsenic = arsenic   # ppm
        self.quartz = quartz     # wt%
        self.carbonates = carbonates # wt%
        self.bwi = bwi           # kWh/t
        self.fes2 = fes2         # wt% pyrite (FeS2) — drives exothermic self-heating

class KeterGeologistAgent:
    def __init__(self, cluster: CarlinOreCluster, target_au: float = 10.0):
        self.cluster = cluster
        self.target_au = target_au
        # Initial MCDA Weights (must sum to 1.0)
        self.weights = {
            "Au": 0.40,
            "BWI": 0.20,
            "TOC": 0.15,
            "As": 0.25
        }
        self.discount_factor = 0.99  # delta_G
        
    def calculate_utility(self, blend_ratio: float, oxide_cluster: CarlinOreCluster) -> float:
        """Calculates Keter's MCDA utility for a given blend ratio (1.0 = 100% Refractory)."""
        # Linear Geochemical Dilution
        f_ref = blend_ratio
        f_ox = 1.0 - blend_ratio
        
        au = f_ref * self.cluster.au + f_ox * oxide_cluster.au
        bwi = f_ref * self.cluster.bwi + f_ox * oxide_cluster.bwi
        toc = f_ref * self.cluster.toc + f_ox * oxide_cluster.toc
        arsenic = f_ref * self.cluster.arsenic + f_ox * oxide_cluster.arsenic
        
        # Attribute Utilities [0, 1]
        u_au = min(1.0, au / max(0.1, self.target_au))
        u_bwi = 1.0 - max(0.0, min(1.0, (bwi - 8.0) / 12.0))      # penalize high BWI
        u_toc = 1.0 - max(0.0, min(1.0, (toc - 0.5) / 3.5))      # penalize high TOC
        u_as = 1.0 - max(0.0, min(1.0, (arsenic - 100.0) / 1100.0)) # penalize arsenic
        
        weighted_utility = (
            self.weights["Au"] * u_au +
            self.weights["BWI"] * u_bwi +
            self.weights["TOC"] * u_toc +
            self.weights["As"] * u_as
        )
        return weighted_utility

    def concede(self, concession_factor: float = 0.05):
        """Concession mechanism: relaxes target Au and adjusts weights to absorb metallurgical limits."""
        self.target_au = max(4.0, self.target_au - 0.5)
        
        # Concede weight from Au to process/environmental parameters
        transferred = self.weights["Au"] * concession_factor
        self.weights["Au"] -= transferred
        self.weights["TOC"] += transferred * 0.5
        self.weights["As"] += transferred * 0.5
        logger.info(f"[Keter Concession] New target Au: {self.target_au:.2f} | Weights: {self.weights}")


class TeomimMetallurgistAgent:
    def __init__(self, init_pri_thresh: float = 1.5, init_emissions_limit: float = 0.45):
        self.pri_threshold = init_pri_thresh
        self.emissions_limit = init_emissions_limit
        self.discount_factor = 0.98  # delta_M

    # Technically preg-robbing index is described by non-linear isotherms BUT at this point linear additive model is a reasonable simplification for
    # this particular problem of physical chemistry (must be corrected first for the real pilot)         
    def calculate_pri(self, toc: float, tcm: float) -> float:
        """Preg-Robbing Index: PRI = 1.2·TOC + 0.8·TCM (linear additive model).
        
        TOC (Total Organic Carbon) drives gold adsorption via activated kerogen.
        TCM (Total Carbonate Mineral) buffers pH during CIL leaching, amplifying preg-robbing.
        Coefficients calibrated for Carlin-type double-refractory ores (Nevada).
        """
        return 1.2 * toc + 0.8 * tcm

    def simulate_roasting(self, blend_ratio: float, ref_cluster: CarlinOreCluster, 
                          ox_cluster: CarlinOreCluster, control_params: Dict[str, float]) -> Tuple[float, float, float]:
        """
        Thermodynamic simulation of roasting outcomes.
        Returns: (Gold Recovery %, As2O3 emissions mg/Nm3, Sintering risk [0..1])
        """
        f_ref = blend_ratio
        f_ox = 1.0 - blend_ratio
        
        # Diluted geochemistry
        toc = f_ref * ref_cluster.toc + f_ox * ox_cluster.toc
        tcm = f_ref * ref_cluster.tcm + f_ox * ox_cluster.tcm
        arsenic = f_ref * ref_cluster.arsenic + f_ox * ox_cluster.arsenic
        carbonates = f_ref * ref_cluster.carbonates + f_ox * ox_cluster.carbonates
        quartz = f_ref * ref_cluster.quartz + f_ox * ox_cluster.quartz
        fes2 = f_ref * ref_cluster.fes2 + f_ox * ox_cluster.fes2
        
        # Control parameters
        f_rate = control_params.get("feed_rate_tph", 100.0)
        excess_air = control_params.get("excess_air_pct", 30.0)
        
        # Compute wall temperature dynamically from exothermic self-heating
        # (simplified version of compute_physics wall temp equation)
        convective_base = 550.0
        radiant = 8.0  # gas fuel default (0.2 luminosity * 40 scale)
        exo_gain = 12.0 * fes2  # exothermic self-heating from pyrite oxidation
        excess_air_cooling = (excess_air - 30.0) * 2.0
        t_wall = convective_base + exo_gain + radiant - excess_air_cooling
        t_wall = max(450.0, min(800.0, t_wall))
        
        # 1. PRI and Base Recovery (Preg-robbing impact)
        pri = self.calculate_pri(toc, tcm)
        r_base = max(10.0, 85.0 * (1.0 - 0.15 * pri))
        
        # 2. Roasting Oxidation Efficiency
        t_ign = 550.0  # Sulfide ignition temp
        eta_roast = 0.0
        if t_wall > t_ign:
            residence_factor = 150.0 / max(1.0, f_rate)
            eta_roast = 1.0 - np.exp(-0.02 * (t_wall - t_ign) * residence_factor * (1.0 / (1.0 + 0.2 * pri)))
            
        r_roast = r_base + (100.0 - r_base) * eta_roast
        
        # 3. Porosity model: sintering above 700°C closes pores, killing cyanide penetration
        t_sinter_crit_por = 700.0
        por_slope = 0.15
        por_ref_time = 5.0
        recovery_loss_per_por = 0.8
        residence_time = 150.0 / max(1.0, f_rate)  # seconds estimate
        
        temp_diff = t_wall - t_sinter_crit_por
        temp_factor = 1.0 / (1.0 + np.exp(-temp_diff * por_slope))
        time_factor = min(2.0, residence_time / por_ref_time)
        porosity_loss = min(1.0, temp_factor * time_factor)
        r_roast = r_roast * (1.0 - porosity_loss * recovery_loss_per_por)
        
        # 4. Arsenic Volatilization Stack Emissions
        base_emissions = 0.0006 * arsenic * np.exp(-0.06 * excess_air) * (f_rate / 100.0)
        
        entrainment_carryover = 0.0
        opt_air = 33.0
        if excess_air > opt_air:
            gamma = 0.0045
            entrainment_carryover = (0.0006 * arsenic * (f_rate / 100.0)) * gamma * ((excess_air - opt_air) ** 2)
            
        emissions = base_emissions + entrainment_carryover
        
        # 5. Clay Sintering Risk
        t_sinter_crit = 710.0 - 6.0 * carbonates + 2.0 * quartz
        sinter_risk = 1.0 / (1.0 + np.exp(-0.15 * (t_wall - t_sinter_crit)))
        
        return r_roast, emissions, sinter_risk

    def calculate_utility(self, blend_ratio: float, ref_cluster: CarlinOreCluster, 
                          ox_cluster: CarlinOreCluster, control_params: Dict[str, float]) -> float:
        """Computes Teomim's metallurgical & process utility."""
        toc = blend_ratio * ref_cluster.toc + (1.0 - blend_ratio) * ox_cluster.toc
        tcm = blend_ratio * ref_cluster.tcm + (1.0 - blend_ratio) * ox_cluster.tcm
        pri = self.calculate_pri(toc, tcm)
        
        r_roast, emissions, sinter_risk = self.simulate_roasting(blend_ratio, ref_cluster, ox_cluster, control_params)
        
        # Utility begins with roasting gold recovery percentage
        utility = r_roast
        
        # Environmental/Physical Constraint Penalties
        if emissions > self.emissions_limit:
            utility -= 80.0 * (emissions - self.emissions_limit)
        if sinter_risk > 0.15:
            utility -= 50.0 * (sinter_risk - 0.15)
        if pri > self.pri_threshold:
            utility -= 30.0 * (pri - self.pri_threshold)
            
        return max(0.0, utility)

    def concede(self, control_params: Dict[str, float]):
        """
        Concession mechanism using process parameters.
        Adjusts PRI and emission thresholds if mitigation controls (Excess Air and Feed Rate) are optimized.
        """
        # If excess air is high, arsenic is stable. We can absorb slightly more arsenic.
        if control_params.get("excess_air_pct", 30.0) >= 34.0:
            self.emissions_limit = min(0.50, self.emissions_limit + 0.02)
            logger.info(f"[Teomim Concession] Relaxed As2O3 limit to: {self.emissions_limit:.3f} mg/Nm3 (due to high excess air).")
            
        # If feed rate is reduced, calcination residence time is high. We can handle higher preg-robbing indices.
        if control_params.get("feed_rate_tph", 100.0) <= 95.0:
            self.pri_threshold += 0.2
            logger.info(f"[Teomim Concession] Relaxed PRI threshold to: {self.pri_threshold:.2f} (due to high residence time).")


class GameTheoreticBargainingEngine:
    def __init__(self, geologist: KeterGeologistAgent, metallurgist: TeomimMetallurgistAgent,
                 oxide_cluster: CarlinOreCluster):
        self.geologist = geologist
        self.metallurgist = metallurgist
        self.oxide_cluster = oxide_cluster
        
        # Disagreement Threat Points
        self.d_G = 0.15  # Low recovery via raw low-grade atmospheric leaching
        self.d_M = 60.0  # Operating on 100% clean oxide (safe, but low grade)

    def negotiate(self, max_turns: int = 10) -> Dict[str, Any]:
        """Runs the Rubinstein Alternating-Offers game dynamically adjusting variables via a bilateral Bisection search."""
        logger.info("Initializing Game-Theoretic Bisection Negotiation Loop...")
        
        # Bisection search boundaries
        f_low = 0.0
        f_high = 1.0
        f_ref_proposal = 1.0
        
        # Default control settings for roaster
        # Note: wall_temp_c is no longer a control dial — it's computed dynamically
        # from exothermic reaction heat and air staging in simulate_roasting()
        control_params = {
            "feed_rate_tph": 100.0,
            "excess_air_pct": 30.0
        }
        
        best_accepted_f = 0.0
        best_control_params = control_params.copy()
        success = False
        
        history = []
        
        for turn in range(max_turns):
            # Calculate discount weights for delay costs
            discount_g = self.geologist.discount_factor ** turn
            discount_m = self.metallurgist.discount_factor ** turn
            
            # --- TURN A: Geologist Proposes ---
            u_g = self.geologist.calculate_utility(f_ref_proposal, self.oxide_cluster) * discount_g
            u_m = self.metallurgist.calculate_utility(f_ref_proposal, self.geologist.cluster, self.oxide_cluster, control_params) * discount_m
            
            r_roast, emissions, sinter_risk = self.metallurgist.simulate_roasting(
                f_ref_proposal, self.geologist.cluster, self.oxide_cluster, control_params
            )
            
            # Check acceptance condition for Metallurgist
            is_accepted = u_m > self.d_M and emissions <= self.metallurgist.emissions_limit and sinter_risk <= 0.15
            
            history.append({
                "turn": turn,
                "offerer": "Geologist",
                "proposed_ref_ratio": f_ref_proposal,
                "u_G": u_g,
                "u_M": u_m,
                "roast_recovery": r_roast,
                "emissions": emissions,
                "sinter_risk": sinter_risk
            })
            
            if is_accepted:
                logger.info(f"Consensus achieved at Turn {turn}! Proposal {f_ref_proposal*100:.1f}% Refractory ore is feasible.")
                best_accepted_f = f_ref_proposal
                best_control_params = control_params.copy()
                success = True
                
                # Since accepted, we can try to push higher refractory ore (more profit/grade)
                f_low = f_ref_proposal
            else:
                # Dynamic Rejection Reason Construction
                rejection_reasons = []
                
                toc = f_ref_proposal * self.geologist.cluster.toc + (1.0 - f_ref_proposal) * self.oxide_cluster.toc
                tcm = f_ref_proposal * self.geologist.cluster.tcm + (1.0 - f_ref_proposal) * self.oxide_cluster.tcm
                pri = self.metallurgist.calculate_pri(toc, tcm)
                
                if u_m <= self.d_M:
                    if pri > self.metallurgist.pri_threshold:
                        rejection_reasons.append(f"Low utility ({u_m:.2f} <= limit {self.d_M:.1f}) due to high Preg-Robbing Index (PRI={pri:.2f} > limit {self.metallurgist.pri_threshold:.2f})")
                    else:
                        rejection_reasons.append(f"Low utility ({u_m:.2f} <= limit {self.d_M:.1f})")
                if emissions > self.metallurgist.emissions_limit:
                    rejection_reasons.append(f"Emissions spike (emissions={emissions:.3f} > limit {self.metallurgist.emissions_limit:.3f} mg/Nm3)")
                if sinter_risk > 0.15:
                    rejection_reasons.append(f"Sintering Risk ({sinter_risk*100:.1f}% > limit 15.0%)")
                
                rejection_reason_str = "; ".join(rejection_reasons)
                logger.warning(f"Turn {turn}: Metallurgist REJECTS {f_ref_proposal*100:.1f}% Refractory. Reason: {rejection_reason_str}")
                
                # Concede and mitigate: Metallurgist scales down feed and boosts air to stabilize kiln
                if emissions > self.metallurgist.emissions_limit:
                    control_params["excess_air_pct"] = min(40.0, control_params["excess_air_pct"] + 2.0)
                if sinter_risk > 0.15:
                    # Reduce feed rate to lower exothermic heat and increase quench air
                    control_params["feed_rate_tph"] = max(85.0, control_params["feed_rate_tph"] - 3.0)
                    control_params["excess_air_pct"] = min(45.0, control_params["excess_air_pct"] + 1.5)
                    
                self.metallurgist.concede(control_params)
                
                # Keter concedes: accepts slightly lower grades and absorbs penalties
                self.geologist.concede()
                
                # Since rejected, we must reduce the refractory ore content
                f_high = f_ref_proposal
            
            # Check convergence
            if f_high - f_low < 0.01:
                logger.info(f"Bisection search converged with tolerance < 0.01 at Turn {turn}.")
                break
                
            # New proposal is bisected
            f_ref_proposal = max(0.05, (f_low + f_high) / 2.0)
            
        if success:
            logger.info(f"Final consensus achieved: {best_accepted_f*100:.1f}% Refractory ore.")
            return self._finalize_deal(best_accepted_f, best_control_params, history, success=True)
        else:
            logger.error("Negotiation reached maximum rounds without clean convergence. Triggering Nash threat fallback.")
            return self._finalize_deal(0.0, control_params, history, success=False)

    def _finalize_deal(self, final_ref_ratio: float, control_params: Dict[str, float], 
                       history: List[Dict[str, Any]], success: bool) -> Dict[str, Any]:
        
        r_roast, emissions, sinter_risk = self.metallurgist.simulate_roasting(
            final_ref_ratio, self.geologist.cluster, self.oxide_cluster, control_params
        )
        
        toc = final_ref_ratio * self.geologist.cluster.toc + (1.0 - final_ref_ratio) * self.oxide_cluster.toc
        tcm = final_ref_ratio * self.geologist.cluster.tcm + (1.0 - final_ref_ratio) * self.oxide_cluster.tcm
        pri = self.metallurgist.calculate_pri(toc, tcm)
        
        return {
            "converged": success,
            "final_refractory_ratio": final_ref_ratio,
            "final_oxide_ratio": 1.0 - final_ref_ratio,
            "optimized_recovery": r_roast,
            "stack_emissions_mg_nm3": emissions,
            "sintering_risk": sinter_risk,
            "feed_pri": pri,
            "control_settings": control_params,
            "negotiation_history": history
        }
