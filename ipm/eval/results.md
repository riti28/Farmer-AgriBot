# Ask IPM evaluation: OLD vs NEW

- NEW: BAAI/bge-small-en-v1.5 embeddings (weight 0.7) + TF-IDF (weight 0.3), synonyms, threshold 0.47, 49 notes, 114 chunks. No LLM.
- OLD: TF-IDF only, threshold 0.15.
- Honest caveats: we wrote both question sets ourselves, the new notes were written with
  everyday farmer problems in mind, and the threshold was chosen by looking at these same
  questions. Real farmer questions would score lower. 'Grounded' only checks one key word.

## Summary

| Question set | System | Hit@3 | Answered | Off-topic refused | Grounded (all) |
|---|---|---|---|---|---|
| Original questions (27 + 3 off-topic) | OLD: TF-IDF, 29 notes | 1.0 | 1.0 | 3/3 | 0.97 |
| Original questions (27 + 3 off-topic) | OLD + new notes: TF-IDF, 49 notes | 1.0 | 1.0 | 3/3 | 0.97 |
| Original questions (27 + 3 off-topic) | NEW: hybrid + synonyms, 49 notes | 1.0 | 1.0 | 3/3 | 0.87 |
| Farmer questions (25 + 3 off-topic) | OLD: TF-IDF, 29 notes | 0.32 | 0.4 | 3/3 | 0.29 |
| Farmer questions (25 + 3 off-topic) | OLD + new notes: TF-IDF, 49 notes | 0.92 | 0.68 | 3/3 | 0.68 |
| Farmer questions (25 + 3 off-topic) | NEW: hybrid + synonyms, 49 notes | 0.96 | 1.0 | 3/3 | 0.79 |

## Threshold check (NEW system, 52 answerable + 6 off-topic questions)

| threshold | answerable questions answered | off-topic questions refused |
|---|---|---|
| 0.4 | 52 | 4 |
| 0.45 | 52 | 5 |
| 0.46 | 52 | 6 |
| 0.47 | 52 | 6 |
| 0.48 | 51 | 6 |
| 0.5 | 51 | 6 |
| 0.55 | 46 | 6 |
| 0.6 | 39 | 6 |

## NEW system, per question: Original questions

| question | top3 | hit | answered | grounded |
|---|---|---|---|---|
| What is integrated pest management? | ipm01_what_is_ipm, ipm01_what_is_ipm, ipm27_pesticide_code | True | True | True |
| What is push-pull intercropping against fall armyworm? | ipm26_faw_africa, ipm09_faw_karnataka, ipm08_faw_management | True | True | True |
| What is the economic threshold level? | ipm03_economic_threshold, ipm03_economic_threshold, ipm38_when_to_spray | True | True | True |
| How should I scout my field for pests? | ipm03_economic_threshold, ipm24_natural_enemies, ipm38_when_to_spray | True | True | False |
| What is hopper burn in rice? | ipm04_rice_bph, ipm46_rice_pests_overview, ipm34_jassids | True | True | True |
| At how many hoppers per hill should I act against brown planthopper? | ipm04_rice_bph, ipm04_rice_bph, ipm25_rice_global | True | True | True |
| What are the symptoms of yellow stem borer: dead heart and white ear? | ipm05_rice_stem_borer, ipm05_rice_stem_borer, ipm39_yellow_leaves | True | True | True |
| How can I reduce rice stem borer egg masses before transplanting? | ipm05_rice_stem_borer, ipm05_rice_stem_borer, ipm05_rice_stem_borer | True | True | True |
| How do I manage rice leaf folder? | ipm06_rice_other_pests, ipm06_rice_other_pests, ipm06_rice_other_pests | True | True | True |
| How do I identify fall armyworm larva? | ipm07_faw_identification, ipm09_faw_karnataka, ipm07_faw_identification | True | True | True |
| When was fall armyworm first reported in India? | ipm07_faw_identification, ipm09_faw_karnataka, ipm26_faw_africa | True | True | True |
| Can sand or ash in the maize whorl control fall armyworm? | ipm08_faw_management, ipm08_faw_management, ipm26_faw_africa | True | True | True |
| How many pheromone traps per acre for fall armyworm in Karnataka? | ipm09_faw_karnataka, ipm18_traps, ipm18_traps | True | True | True |
| How was pink bollworm eradicated in the United States? | ipm29_pbw_usa, ipm29_pbw_usa, ipm11_pbw_management | True | True | False |
| What damage does pink bollworm cause in cotton bolls? | ipm10_pbw_biology, ipm10_pbw_biology, ipm11_pbw_management | True | True | False |
| Is Bt cotton still protecting against pink bollworm? | ipm10_pbw_biology, ipm29_pbw_usa, ipm10_pbw_biology | True | True | True |
| Why should cotton crop be terminated by December? | ipm11_pbw_management, ipm12_cotton_raichur, ipm49_cotton_pests_overview | True | True | True |
| What is mating disruption for pink bollworm? | ipm11_pbw_management, ipm29_pbw_usa, ipm10_pbw_biology | True | True | True |
| How to manage sucking pests of cotton in north Karnataka? | ipm12_cotton_raichur, ipm49_cotton_pests_overview, ipm12_cotton_raichur | True | True | True |
| What weather favours early shoot borer in sugarcane? | ipm13_esb_biology, ipm14_esb_management, ipm14_esb_management | True | True | True |
| Does trash mulching reduce sugarcane early shoot borer? | ipm14_esb_management, ipm13_esb_biology, ipm14_esb_management | True | True | True |
| What are the symptoms of neck blast in ragi? | ipm15_ragi_blast, ipm15_ragi_blast, ipm16_ragi_blast_management | True | True | True |
| Which ragi variety is resistant to blast? | ipm16_ragi_blast_management, ipm15_ragi_blast, ipm28_finger_millet_africa | True | True | True |
| What is Trichogramma and how does it work? | ipm41_trichoderma, ipm41_trichoderma, ipm14_esb_management | True | True | False |
| How often should pheromone lures be changed? | ipm18_traps, ipm03_economic_threshold, ipm18_traps | True | True | True |
| What safety precautions should I take while spraying pesticide? | ipm19_safe_pesticide_use, ipm38_when_to_spray, ipm38_when_to_spray | True | True | True |
| Where can a farmer get a diagnosis for a sick plant? | ipm20_extension_plant_clinics, ipm45_crop_dying, ipm20_extension_plant_clinics | True | True | True |
| What is the price of gold in Delhi today? | ipm28_finger_millet_africa, ipm03_economic_threshold, ipm44_pesticide_label | - | - | True |
| Who won the cricket world cup? | ipm02_ipm_in_india, ipm27_pesticide_code, ipm44_pesticide_label | - | - | True |
| How do I repair my smartphone screen? | ipm28_finger_millet_africa, ipm15_ragi_blast, ipm28_finger_millet_africa | - | - | True |

## NEW system, per question: Farmer questions

| question | top3 | hit | answered | grounded |
|---|---|---|---|---|
| worms eating my maize | ipm47_maize_pests_overview, ipm47_maize_pests_overview, ipm08_faw_management | True | True | False |
| which spray for cotton | ipm49_cotton_pests_overview, ipm49_cotton_pests_overview, ipm42_bt | True | True | True |
| insects in sugarcane | ipm48_sugarcane_pests_overview, ipm48_sugarcane_pests_overview, ipm48_sugarcane_pests_overview | True | True | False |
| when should I spray insecticide | ipm38_when_to_spray, ipm38_when_to_spray, ipm38_when_to_spray | True | True | False |
| rice crop pest | ipm46_rice_pests_overview, ipm25_rice_global, ipm46_rice_pests_overview | True | True | True |
| my rice leaves are turning yellow what to do | ipm39_yellow_leaves, ipm39_yellow_leaves, ipm05_rice_stem_borer | True | True | True |
| how to control aphids | ipm30_aphids, ipm30_aphids, ipm30_aphids | True | True | True |
| my crop is dying | ipm45_crop_dying, ipm45_crop_dying, ipm41_trichoderma | True | True | True |
| what is neem oil | ipm40_neem, ipm40_neem, ipm40_neem | True | True | False |
| small white flies under my cotton leaves | ipm31_whitefly, ipm31_whitefly, ipm31_whitefly | True | True | True |
| white ants are eating my sugarcane | ipm35_termites, ipm48_sugarcane_pests_overview, ipm48_sugarcane_pests_overview | True | True | True |
| rats in my paddy field | ipm37_rats, ipm37_rats, ipm37_rats | True | True | True |
| my plant leaves have silver streaks | ipm06_rice_other_pests, ipm39_yellow_leaves, ipm39_yellow_leaves | False | True | False |
| tiny red insects with web under the leaves | ipm32_thrips, ipm33_mites, ipm30_aphids | True | True | True |
| cotton leaf edges turning red and curling | ipm31_whitefly, ipm34_jassids, ipm34_jassids | True | True | False |
| grubs eating roots of groundnut | ipm36_white_grub, ipm36_white_grub, ipm36_white_grub | True | True | True |
| how much pesticide per acre | ipm44_pesticide_label, ipm44_pesticide_label, ipm10_pbw_biology | True | True | True |
| is there an organic way to kill caterpillars | ipm08_faw_management, ipm24_natural_enemies, ipm40_neem | True | True | True |
| how do I treat seeds before sowing | ipm43_seed_treatment, ipm43_seed_treatment, ipm28_finger_millet_africa | True | True | True |
| what is trichoderma | ipm41_trichoderma, ipm41_trichoderma, ipm43_seed_treatment | True | True | True |
| how to stop fungus disease in roots | ipm41_trichoderma, ipm41_trichoderma, ipm45_crop_dying | True | True | True |
| black spots on ragi leaves | ipm15_ragi_blast, ipm15_ragi_blast, ipm28_finger_millet_africa | True | True | True |
| my paddy has brown dried patches | ipm04_rice_bph, ipm46_rice_pests_overview, ipm45_crop_dying | True | True | True |
| worms inside cotton bolls | ipm10_pbw_biology, ipm10_pbw_biology, ipm47_maize_pests_overview | True | True | True |
| what does the colour on a pesticide bottle mean | ipm44_pesticide_label, ipm19_safe_pesticide_use, ipm27_pesticide_code | True | True | True |
| how do I apply for a crop loan | ipm21_farmer_field_schools, ipm35_termites, ipm43_seed_treatment | - | - | True |
| what is the price of tomato today | ipm44_pesticide_label, ipm09_faw_karnataka, ipm10_pbw_biology | - | - | True |
| how to repair my tractor engine | ipm08_faw_management, ipm09_faw_karnataka, ipm44_pesticide_label | - | - | True |
