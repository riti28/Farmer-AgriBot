# RAG evaluation results

- Retriever: TF-IDF + cosine similarity, top-3, threshold 0.15
- Corpus: 29 notes, 65 chunks
- Note: we wrote these questions from our own notes and picked the threshold by looking at
  them, so the scores are optimistic. Real farmer questions would score lower.

| Metric | Score |
|---|---|
| Hit@3 (27 answerable questions) | 1.00 |
| Grounded-answer score (all 30 questions) | 0.97 |

## Per-question results

| id | question | top-3 doc ids | hit@3 | grounded |
|---|---|---|---|---|
| 1 | What is integrated pest management? | ipm01_what_is_ipm, ipm01_what_is_ipm, ipm04_rice_bph | True | True |
| 2 | What is push-pull intercropping against fall armyworm? | ipm26_faw_africa, ipm26_faw_africa, ipm09_faw_karnataka | True | True |
| 3 | What is the economic threshold level? | ipm03_economic_threshold, ipm03_economic_threshold, ipm02_ipm_in_india | True | True |
| 4 | How should I scout my field for pests? | ipm03_economic_threshold, ipm21_farmer_field_schools, ipm21_farmer_field_schools | True | True |
| 5 | What is hopper burn in rice? | ipm04_rice_bph, ipm25_rice_global, ipm04_rice_bph | True | True |
| 6 | At how many hoppers per hill should I act against brown planthopper? | ipm04_rice_bph, ipm04_rice_bph, ipm04_rice_bph | True | True |
| 7 | What are the symptoms of yellow stem borer: dead heart and white ear? | ipm05_rice_stem_borer, ipm05_rice_stem_borer, ipm13_esb_biology | True | True |
| 8 | How can I reduce rice stem borer egg masses before transplanting? | ipm05_rice_stem_borer, ipm05_rice_stem_borer, ipm05_rice_stem_borer | True | True |
| 9 | How do I manage rice leaf folder? | ipm06_rice_other_pests, ipm06_rice_other_pests, ipm06_rice_other_pests | True | True |
| 10 | How do I identify fall armyworm larva? | ipm07_faw_identification, ipm26_faw_africa, ipm09_faw_karnataka | True | True |
| 11 | When was fall armyworm first reported in India? | ipm07_faw_identification, ipm26_faw_africa, ipm09_faw_karnataka | True | True |
| 12 | Can sand or ash in the maize whorl control fall armyworm? | ipm08_faw_management, ipm08_faw_management, ipm26_faw_africa | True | True |
| 13 | How many pheromone traps per acre for fall armyworm in Karnataka? | ipm09_faw_karnataka, ipm18_traps, ipm18_traps | True | True |
| 14 | How was pink bollworm eradicated in the United States? | ipm29_pbw_usa, ipm29_pbw_usa, ipm10_pbw_biology | True | True |
| 15 | What damage does pink bollworm cause in cotton bolls? | ipm10_pbw_biology, ipm10_pbw_biology, ipm11_pbw_management | True | True |
| 16 | Is Bt cotton still protecting against pink bollworm? | ipm10_pbw_biology, ipm29_pbw_usa, ipm10_pbw_biology | True | True |
| 17 | Why should cotton crop be terminated by December? | ipm11_pbw_management, ipm10_pbw_biology, ipm12_cotton_raichur | True | True |
| 18 | What is mating disruption for pink bollworm? | ipm11_pbw_management, ipm29_pbw_usa, ipm10_pbw_biology | True | True |
| 19 | How to manage sucking pests of cotton in north Karnataka? | ipm12_cotton_raichur, ipm12_cotton_raichur, ipm23_cultural_practices | True | True |
| 20 | What weather favours early shoot borer in sugarcane? | ipm13_esb_biology, ipm14_esb_management, ipm14_esb_management | True | True |
| 21 | Does trash mulching reduce sugarcane early shoot borer? | ipm14_esb_management, ipm13_esb_biology, ipm14_esb_management | True | True |
| 22 | What are the symptoms of neck blast in ragi? | ipm15_ragi_blast, ipm16_ragi_blast_management, ipm28_finger_millet_africa | True | True |
| 23 | Which ragi variety is resistant to blast? | ipm16_ragi_blast_management, ipm15_ragi_blast, ipm15_ragi_blast | True | True |
| 24 | What is Trichogramma and how does it work? | ipm17_biocontrol_agents, ipm08_faw_management, ipm14_esb_management | True | False |
| 25 | How often should pheromone lures be changed? | ipm18_traps, ipm18_traps, ipm03_economic_threshold | True | True |
| 26 | What safety precautions should I take while spraying pesticide? | ipm19_safe_pesticide_use, ipm19_safe_pesticide_use, ipm29_pbw_usa | True | True |
| 27 | Where can a farmer get a diagnosis for a sick plant? | ipm20_extension_plant_clinics, ipm02_ipm_in_india, ipm20_extension_plant_clinics | True | True |
| 28 | What is the price of gold in Delhi today? | ipm29_pbw_usa, ipm29_pbw_usa, ipm28_finger_millet_africa | - | True |
| 29 | Who won the cricket world cup? | ipm29_pbw_usa, ipm29_pbw_usa, ipm28_finger_millet_africa | - | True |
| 30 | How do I repair my smartphone screen? | ipm28_finger_millet_africa, ipm29_pbw_usa, ipm29_pbw_usa | - | True |
