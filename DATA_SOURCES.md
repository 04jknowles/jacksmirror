# Data Sources and Assembly Methods

This document describes how the datasets in this project were identified, obtained, extracted, and curated. It is intended to provide full transparency on data provenance for reproducibility and peer review.

---

## 1. Primary Data Source

### Hughes et al. (2008) — Table S1

**Full citation:**
Hughes, W.O.H., Oldroyd, B.P., Beekman, M. & Ratnieks, F.L.W. (2008). Ancestral monogamy shows kin selection is key to the evolution of eusociality. *Science*, 320(5880), 1213-1216. DOI: [10.1126/science.1156108](https://doi.org/10.1126/science.1156108)

**What it contains:**
The Supporting Online Material (SOM) for this paper includes Table S1, a comprehensive dataset of female mating frequencies and worker totipotency across 267 species of eusocial Hymenoptera (bees, wasps, and ants). This is the most widely cited comparative dataset for mating frequency in social insects.

**How we obtained it:**
The SOM PDF was downloaded from the *Science* journal website at:
`www.sciencemag.org/cgi/content/full/320/5880/1213/DC1`

**How we extracted the data:**
Table S1 spans pages 8-20 of the 20-page SOM PDF (`hughes.som.pdf`). The table was extracted programmatically using PyMuPDF (v1.25), a Python library for PDF parsing. The extraction process:

1. Opened the PDF with `fitz.open()`
2. Used PyMuPDF's `page.find_tables()` method to detect tabular structures on each page
3. Extracted cell values from the detected tables using `table.extract()`
4. Parsed each row into structured fields: family, genus, species, proportion of females mating multiply, effective mating frequency, polyandry level, references, worker totipotency, and totipotency references
5. Filtered header rows and empty rows
6. Wrote the structured data to CSV

**Result:** `data/hughes_2008_table_s1.csv` — 266 species (one species fewer than the stated 267, likely due to a row spanning a page boundary in the PDF).

**Columns in the extracted dataset:**

| Column | Description | Source |
|--------|-------------|--------|
| `family` | Taxonomic family (Apidae, Halictidae, Vespidae, Formicidae, Sphecidae) | Table S1 |
| `genus` | Genus name | Table S1 |
| `species` | Species epithet | Table S1 |
| `prop_females_poly` | Proportion of females mating with more than one male (0-1, or "?" if unknown) | Table S1, from molecular markers or behavioural observations |
| `effective_mating_freq` | Effective mating frequency (number of effective mates per queen) | Table S1, computed from paternity data |
| `polyandry_level` | Categorical: 0 = monandry, 1 = low polyandry (>1 but <2 effective mates), 2 = high polyandry (>2 effective mates) | Table S1 |
| `refs` | Reference codes for mating frequency data (e.g., "(S33, 34)") | Table S1 |
| `worker_totipotency` | Whether workers can mate and reproduce sexually: "yes", "no", or "n/a" | Table S1 |
| `totipotency_refs` | Reference codes for totipotency data | Table S1 |

**Data quality notes:**
- For 81% of species, mating frequency data were from molecular markers (allozymes or microsatellites). For 19%, data were from behavioural observations only (marked as uncertain in the original table with shading).
- Some cells contain "?" indicating the value was not determined.
- Some effective mating frequency values contain ">1" rather than a precise number.
- The original table marks uncertain species with shading; this distinction is not preserved in the CSV extraction. Hughes et al. ran parallel analyses with and without these uncertain species and found consistent results.

---

## 2. Curated Analysis Dataset

### bee_species.csv

**What it contains:**
A curated dataset of 70 bee species designed to test the monogamy hypothesis. Unlike the Hughes Table S1 (which contains only eusocial species), this dataset includes the full spectrum of social organisation in bees: solitary, communal, semisocial, primitively eusocial, and advanced eusocial species.

**Why a separate dataset was needed:**
The Hughes et al. Table S1 contains only eusocial Hymenoptera (bees, wasps, and ants). To test whether monandry is *associated* with eusociality (rather than simply describing mating variation within eusocial species), we need a dataset that includes both eusocial and non-eusocial species for comparison. This curated dataset provides that contrast.

**How it was assembled:**

The dataset was compiled from multiple published sources, with the following assembly process:

1. **Eusocial species (n=28):** Species and mating frequency data were drawn primarily from Hughes et al. (2008) Table S1, supplemented by the original references cited therein. This includes:
   - 14 advanced eusocial species: 4 *Apis* (honeybees, polyandrous), 10 Meliponini (stingless bees, monandrous)
   - 14 primitively eusocial species: 8 *Bombus* (bumblebees), 4 Halictini (sweat bees), 2 Augochlorini

2. **Non-eusocial species (n=42):** Species were selected to represent the diversity of non-eusocial bee lineages:
   - 35 solitary species across 6 families (Apidae, Megachilidae, Andrenidae, Colletidae, Halictidae, Melittidae)
   - 6 communal species (Allodapini, Euglossini, Halictini)
   - 1 semisocial species (*Lasioglossum hemichalceum*)

3. **Cross-referencing with Hughes et al. (2008):** For the 21 species present in both the curated dataset and Hughes Table S1, effective mating frequency values were updated to match the Hughes primary source. Nine species had values that differed from the originally compiled secondary sources:

   | Species | Original value | Hughes value | Original source |
   |---------|---------------|-------------|-----------------|
   | *Apis mellifera* | 12.0 | 11.6 | Secondary compilation |
   | *Apis cerana* | 10.0 | 14.1 | Secondary compilation |
   | *Apis dorsata* | 17.0 | 44.2 | Oldroyd et al. 1996 |
   | *Apis florea* | 7.5 | 7.9 | Palmer et al. 2002 |
   | *Melipona beecheii* | 1.0 | 1.16 | Peters et al. 1999 |
   | *Austroplebeia symei* | 1.0 | 1.06 | Green & Oldroyd 2002 |
   | *Bombus impatiens* | 1.0 | 1.13 | Hines et al. 2004 |
   | *Bombus hypnorum* | 1.4 | 1.12 | Paxton et al. 2001 |
   | *Halictus ligatus* | 1.0 | 2.9 | Richards et al. 2003 |

**Columns in the curated dataset:**

| Column | Description | How determined |
|--------|-------------|---------------|
| `species` | Binomial species name | Published taxonomy |
| `family` | Taxonomic family | Published taxonomy (Michener 2007) |
| `tribe` | Taxonomic tribe | Published taxonomy |
| `sociality` | Social organisation level | Published classifications (see below) |
| `mating_system` | "monandrous" or "polyandrous" | Based on published mating frequency data |
| `effective_mates` | Mean effective number of mates per queen | Hughes et al. 2008 where available; otherwise from cited sources |
| `haplodiploidy` | Haplodiploid sex determination (True for all bees) | Universal in Hymenoptera |
| `colony_size` | Approximate colony size (workers) | Published estimates; blank for non-colonial species |
| `source` | Primary citation for mating system data | See source list below |

**Sociality classification criteria:**

| Level | Definition | Criteria |
|-------|-----------|----------|
| solitary | No cooperative brood care | Each female provisions her own nest independently |
| communal | Shared nest, no division of labour | Females share a nest entrance but each provisions her own cells |
| semisocial | Division of labour within a generation | Workers and reproductives of the same generation |
| primitively eusocial | Overlapping generations, reversible castes | Queen-worker distinction exists but workers retain reproductive capacity |
| advanced eusocial | Overlapping generations, irreversible castes | Morphologically distinct queens and workers; workers cannot mate |

Classifications follow Michener (2007) and the sources cited for each species.

**Mating system classification criteria:**
- **Monandrous:** Queens mate with a single male (effective mating frequency approximately 1.0)
- **Polyandrous:** Queens mate with multiple males (effective mating frequency substantially greater than 1.0)

For the 5 polyandrous species in the dataset (all *Apis* plus *Bombus hypnorum*), polyandry is supported by molecular paternity analyses. The *Bombus hypnorum* classification follows Paxton et al. (2001), though Hughes et al. (2008) report a lower effective mating frequency (1.12) with polyandry level 1 (facultative low polyandry).

---

## 3. Source Bibliography

The following references were used in compiling the curated dataset. The `source` column in `bee_species.csv` indicates the primary reference for each species' mating system data.

- Alcock, J. (1984). Long-term maintenance of size variation in two desert grasshopper populations. *Evolution*, 38, 155-161.
- Alcock, J. (1991). Adaptive mate-guarding by males of *Xylocopa varipuncta*. *Journal of Insect Behavior*, 4, 509-521.
- Alcock, J. (1996). The relation between male body size, fighting, and mating success in *Dawson's burrowing bee*. *Journal of Zoology*, 239, 663-674.
- Augusto, S.C. & Garofalo, C.A. (2009). Nesting biology and social structure of *Euglossa cordata*. *Apidologie*, 40, 222-233.
- Bosch, J. & Kemp, W.P. (2002). Developing and establishing bee species as crop pollinators. *Annals of the Entomological Society of America*, 95, 622-636.
- Cameron, S.A. & Ramirez, S. (2001). Nest architecture and nesting ecology of the orchid bee *Eulaema meriana*. *Journal of the Kansas Entomological Society*, 74, 142-165.
- Danforth, B.N. (2002). Evolution of sociality in a primitively eusocial lineage of bees. *PNAS*, 99, 286-290.
- Estoup, A., Scholl, A., Pouvreau, A. & Solignac, M. (1995). Monoandry and polyandry in bumble bees. *Proceedings of the Royal Society B*, 259, 1-7.
- Green, C.L. & Oldroyd, B.P. (2002). Queen mating frequency and maternity of males in the stingless bee *Trigona carbonaria*. *Insectes Sociaux*, 49, 196-202.
- Hines, H.M., Hendrix, D.L., Dang, P.M. & Cameron, S.A. (2004). Molecular phylogeny and implications for classification of *Bombus*. *Journal of Hymenoptera Research*, 13, 107-111.
- Hughes, W.O.H., Oldroyd, B.P., Beekman, M. & Ratnieks, F.L.W. (2008). Ancestral monogamy shows kin selection is key to the evolution of eusociality. *Science*, 320, 1213-1216.
- Kukuk, P.F. & Schwarz, M.P. (1988). Macrocephalic male bees as functional reproductives and colony defenders. *Pan-Pacific Entomologist*, 64, 131-137.
- Kukuk, P.F., Forbes, S.H., Zahorchak, R., Riddle, A. & Pilgrim, K. (1998). Colony relatedness in a communal halictid bee. *Insectes Sociaux*, 45, 227-234.
- Maeta, Y. & Kitamura, T. (1981). Pollinating efficiency of *Osmia cornifrons*. *Bulletin of Tohoku National Agricultural Experiment Station*, 66, 1-42.
- Michener, C.D. (2007). *The Bees of the World* (2nd ed.). Johns Hopkins University Press.
- Mueller, U.G. (1996). Life history and social evolution of the primitively eusocial bee *Augochlorella striata*. *Journal of the Kansas Entomological Society*, 69, 116-138.
- Oldroyd, B.P., Smolenski, A.J., Cornuet, J.-M. & Crozier, R.H. (1996). Levels of polyandry and intracolonial genetic relationships in *Apis florea*. *Behavioural Ecology and Sociobiology*, 37, 329-335.
- Ordway, E. (1966). Systematics of the bee genus *Augochlorella*. *University of Kansas Science Bulletin*, 46, 509-624.
- Palmer, K.A., Oldroyd, B.P., Quezada-Euan, J.J.G., Paxton, R.J. & May-Itza, W.J. (2002). Paternity frequency and maternity of males in some stingless bee species. *Molecular Ecology*, 11, 2107-2113.
- Paxton, R.J., Thoren, P.A., Tengo, J., Estoup, A. & Pamilo, P. (1996). Mating structure and nestmate relatedness in a communal bee. *Molecular Ecology*, 5, 785-793.
- Paxton, R.J., Ayasse, M., Field, J. & Soro, A. (1999). Mating frequency and nesting behaviour in *Lasioglossum malachurum*. *Behavioural Ecology and Sociobiology*, 46, 35-42.
- Paxton, R.J., Thoren, P.A., Estoup, A. & Tengo, J. (2001). Queen-worker conflict over male production and the sex ratio in a facultatively polyandrous bumblebee, *Bombus hypnorum*. *Molecular Ecology*, 10, 2489-2498.
- Paxton, R.J., Bego, L.R., Shah, M.M. & Mateus, S. (2003). Low mating frequency of queens in the stingless bee *Scaptotrigona postica*. *Behavioural Ecology*, 14, 195-199.
- Peso, M. & Richards, M.H. (2010). Not all who wander are lost: nest fidelity in *Xylocopa virginica*. *Insectes Sociaux*, 57, 131-140.
- Peters, J.M., Queller, D.C., Strassmann, J.E. & Solis, C.R. (1999). Maternity assignment and queen replacement in a social wasp. *Proceedings of the Royal Society B*, 266, 1941-1948.
- Pitts-Singer, T.L. & Cane, J.H. (2011). The alfalfa leafcutting bee, *Megachile rotundata*. *Annual Review of Entomology*, 56, 221-237.
- Rehan, S.M. & Richards, M.H. (2010). Nesting biology and subsociality in *Ceratina calcarata*. *The Canadian Entomologist*, 142, 65-74.
- Richards, M.H., French, D. & Paxton, R.J. (2003). It's good to be queen: classically eusocial colony structure and low worker fitness in an obligately social sweat bee. *Molecular Ecology*, 14, 4123-4133.
- Roberts, R.B. (1969). Biology of the bee genus *Agapostemon*. *University of Kansas Science Bulletin*, 48, 689-719.
- Sakagami, S.F. & Maeta, Y. (1977). Some presumably presocial habits of Japanese *Ceratina* bees. *Kontyû*, 45, 166-180.
- Schwarz, M.P., Silberbauer, L.X. & Hurst, P.S. (1998). Intrinsic and extrinsic factors associated with social evolution in allodapine bees. *Philosophical Transactions of the Royal Society B*, 353, 557-567.
- Seidelmann, K. (2006). Open-cell parasitism shapes maternal investment patterns in the red mason bee *Osmia rufa*. *Behavioural Ecology*, 17, 839-848.
- Severinghaus, L.L., Kurtak, B.H. & Eickwort, G.C. (1981). The reproductive behavior of *Anthidium manicatum*. *Behavioural Ecology and Sociobiology*, 9, 51-58.
- Stark, R.E. (1992). Sex ratio and maternal investment in the multivoltine large carpenter bee *Xylocopa sulcatipes*. *Ecological Entomology*, 17, 160-166.
- Wcislo, W.T. (1997). Behavioural environments of sweat bees (Halictidae) in relation to variability in social organization. *In:* Choe, J.C. & Crespi, B.J. (eds), *The Evolution of Social Behavior in Insects and Arachnids*, pp. 316-332.
- Wcislo, W.T. & Danforth, B.N. (1997). Secondarily solitary: the evolutionary loss of social behavior. *Trends in Ecology and Evolution*, 12, 468-474.
- Yanega, D. (1990). Philopatry and nest founding in a primitively social bee, *Halictus rubicundus*. *Behavioural Ecology and Sociobiology*, 27, 37-42.
- Zimmermann, Y., Roubik, D.W. & Eltz, T. (2009). Species-specific attraction to pheromonal analogues in orchid bees. *Behavioural Ecology and Sociobiology*, 63, 1809-1819.

---

## 4. Data Limitations and Caveats

1. **Taxonomic scope:** The curated dataset is restricted to bees (superfamily Apoidea). The Hughes et al. Table S1 includes ants (Formicidae, n=148) and wasps (Vespidae, n=39) which are not included in the primary analysis but are provided as a reference dataset.

2. **Solitary species mating data:** For many solitary bee species, mating system data come from behavioural observations (Michener 2007) rather than molecular paternity analysis. These species are uniformly classified as monandrous with effective_mates = 1.0, which may slightly overstate the certainty of their classification.

3. **Colony size data:** Colony size estimates are approximate and drawn from various sources. Many non-colonial species have no colony_size value (blank in the CSV). Colony size data should be treated as order-of-magnitude estimates rather than precise counts.

4. **Temporal scope:** The data were compiled from studies published between 1966 and 2011. More recent molecular studies may have revised some mating frequency estimates.

5. **Species selection:** The 42 non-eusocial species were selected to represent taxonomic diversity across bee families, not as an exhaustive or random sample. This purposive sampling should be considered when interpreting the statistical results.

6. **Naming conventions:** Some species names may have been revised since the original publications. The names used here follow the conventions of the cited sources. In particular, some Meliponini genera (e.g., *Trigona* subgenera *Tetragonula*, *Heterotrigona*, *Tetragonilla*) have been elevated to full genus status in some taxonomic treatments; the Hughes Table S1 uses both the older and newer naming conventions.

---

## 5. Reproducibility

All data extraction and curation steps can be reproduced:

1. **Hughes Table S1 extraction:** The PDF extraction script uses PyMuPDF's `find_tables()` API on pages 8-20 of the SOM PDF. The extraction code is documented in the project's development history.

2. **Cross-referencing:** The 9 updated effective mating frequency values (Section 2) can be verified by comparing `data/bee_species.csv` against `data/hughes_2008_table_s1.csv` for matching species names.

3. **All analysis code** is included in the `jacksmirror/` package and the `analysis_notebook.ipynb` notebook, enabling complete reproduction of all statistical results from the raw CSV data.
