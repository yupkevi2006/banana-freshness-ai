# Backward-Looking Decay Modeling & Spoilage Rate Analysis

## 1. Scientific Timeline & Terminal Anchor
- **Terminal Anchor (100% Total Spoilage)**: Defined precisely at **Day 7, Night Session**. At this stage, biological senescence is complete, cellular breakdown is absolute, and volatile organic compound emissions reach saturation.
- **Backward-Looking Aging Framework**: Rather than forward progression, our model evaluates shelf-life and decay state in reverse time from the terminal anchor ($t = 0$ hours remaining at Day 7 Night).

## 2. Percentage Spoilage Threshold Mapping
Moving backward from the terminal anchor, milestone thresholds are mapped as follows:
- **50% Spoilage Threshold**: Reached on **Day 4** (`spoil_50 = 0.0`). Marks the onset of rapid ripening, starch-to-sugar conversion, and measurable peel softening.
- **75% Spoilage Threshold**: Reached on **Day 5** (`spoil_75 = 0.0`). Marks advanced ripening, browning initiation, and notable elevation in respiration rate.
- **90% Spoilage Threshold**: Reached on **Day 6** (`spoil_90 = 0.0`). Over-ripening stage with extensive enzymatic degradation.
- **95% - 100% Spoilage Threshold**: Reached on **Day 7** (`spoil_95 = 0.0` to terminal anchor). Complete decay.

## 3. Rate of Change Analysis (Multimodal Modalities)
- **IR Thermal Signatures**: Temperature differentials ($\Delta T$ between banana surface and ambient environment) remain relatively stable during Days 1–3. As respiration spikes on Days 4–6, internal metabolic heat generation causes non-linear shifts in LW-CNN feature embeddings.
- **Methane & VOC Emissions**: Methane sensor readings (`ppm_left_mean`, `ppm_right_mean`) stay near baseline levels through early green-yellow stages (Days 1–3), showing exponential acceleration starting from Day 4 Evening and peaking sharply at Day 6–7.
