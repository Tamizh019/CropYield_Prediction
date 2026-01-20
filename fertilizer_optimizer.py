"""
🧪 Smart Fertilizer Optimizer
Part of AgriVision v3.0

Calculates optimal fertilizer requirements based on soil analysis.
"""

# Standard NPK requirements per crop (kg/hectare)
CROP_NPK_REQUIREMENTS = {
    "Rice": {"N": 120, "P": 60, "K": 40},
    "Wheat": {"N": 150, "P": 60, "K": 40},
    "Maize": {"N": 150, "P": 75, "K": 40},
    "Cotton": {"N": 150, "P": 60, "K": 60},
    "Sugarcane": {"N": 250, "P": 100, "K": 120},
    "Potato": {"N": 180, "P": 80, "K": 100},
    "Tomato": {"N": 120, "P": 80, "K": 80},
    "Onion": {"N": 100, "P": 50, "K": 50},
    "Groundnut": {"N": 25, "P": 50, "K": 40},
    "Soybean": {"N": 30, "P": 60, "K": 40},
    "Mustard": {"N": 80, "P": 40, "K": 40},
    "Chickpea": {"N": 20, "P": 50, "K": 20},
}

# Fertilizer nutrient content (%)
FERTILIZERS = {
    "Urea": {"N": 46, "P": 0, "K": 0, "cost_per_kg": 6},
    "DAP": {"N": 18, "P": 46, "K": 0, "cost_per_kg": 27},
    "MOP": {"N": 0, "P": 0, "K": 60, "cost_per_kg": 18},
    "NPK_10-26-26": {"N": 10, "P": 26, "K": 26, "cost_per_kg": 25},
    "SSP": {"N": 0, "P": 16, "K": 0, "cost_per_kg": 8},
    "Ammonium_Sulphate": {"N": 21, "P": 0, "K": 0, "cost_per_kg": 10},
}


def calculate_fertilizer(crop, soil_n, soil_p, soil_k, area_hectares=1):
    """
    Calculate optimal fertilizer requirements
    
    Args:
        crop: Crop name
        soil_n, soil_p, soil_k: Current soil NPK levels (kg/ha)
        area_hectares: Field area
    
    Returns:
        dict with fertilizer recommendations
    """
    if crop not in CROP_NPK_REQUIREMENTS:
        return {"error": f"Crop '{crop}' not supported"}
    
    req = CROP_NPK_REQUIREMENTS[crop]
    
    # Calculate deficits
    n_deficit = max(0, req["N"] - soil_n)
    p_deficit = max(0, req["P"] - soil_p)
    k_deficit = max(0, req["K"] - soil_k)
    
    # Calculate fertilizer quantities
    urea_kg = (n_deficit / 0.46) * area_hectares
    dap_kg = (p_deficit / 0.46) * area_hectares
    mop_kg = (k_deficit / 0.60) * area_hectares
    
    # Adjust N for DAP contribution
    n_from_dap = dap_kg * 0.18
    if n_from_dap > 0:
        urea_kg = max(0, urea_kg - (n_from_dap / 0.46))
    
    # Calculate costs
    total_cost = (urea_kg * 6) + (dap_kg * 27) + (mop_kg * 18)
    
    return {
        "success": True,
        "crop": crop,
        "area_hectares": area_hectares,
        "soil_status": {
            "current_N": soil_n,
            "current_P": soil_p,
            "current_K": soil_k,
            "required_N": req["N"],
            "required_P": req["P"],
            "required_K": req["K"]
        },
        "deficits": {
            "N": round(n_deficit, 1),
            "P": round(p_deficit, 1),
            "K": round(k_deficit, 1)
        },
        "recommendations": [
            {"fertilizer": "Urea (46-0-0)", "quantity_kg": round(urea_kg, 1), "cost_inr": round(urea_kg * 6, 2)},
            {"fertilizer": "DAP (18-46-0)", "quantity_kg": round(dap_kg, 1), "cost_inr": round(dap_kg * 27, 2)},
            {"fertilizer": "MOP (0-0-60)", "quantity_kg": round(mop_kg, 1), "cost_inr": round(mop_kg * 18, 2)}
        ],
        "total_cost_inr": round(total_cost, 2),
        "application_tips": [
            "Apply basal dose of DAP and MOP at sowing",
            "Split Urea into 2-3 doses during crop growth",
            "Irrigate after fertilizer application"
        ]
    }


def get_supported_crops():
    return list(CROP_NPK_REQUIREMENTS.keys())


if __name__ == "__main__":
    print("🧪 Fertilizer Calculator Test")
    result = calculate_fertilizer("Rice", soil_n=40, soil_p=20, soil_k=15, area_hectares=2)
    print(f"Crop: {result['crop']}")
    for rec in result['recommendations']:
        print(f"  {rec['fertilizer']}: {rec['quantity_kg']} kg (₹{rec['cost_inr']})")
    print(f"Total Cost: ₹{result['total_cost_inr']}")
