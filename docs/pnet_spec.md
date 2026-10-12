# PnET-Succession cohort ecophysiology: implementation spec (v2 library code is the source of truth)

## 0. Sources, citation convention, and precedence

Precedence: **the C# code (Library-PnET-Cohort v2, 2026-09) overrides the PDFs.** Where PDF and code disagree this is flagged "DISAGREE" and listed in Section 9.

Cite tags:
- `[Cohort.cs:N]`, `[SiteCohorts.cs:N]`, `[Hydrology.cs:N]`, `[EPV.cs:N]` (= EcoregionPnETVariables.cs), `[SR.cs:N]` (= PressureHeadSaxton_Rawls.cs), `[SpeciesPnET.cs:N]`, `[CohortData.cs:N]`, `[EstProb.cs:N]`, `[Constants.cs:N]`: all under `/workspace/LandisII/ref_pnet/Library-PnET-Cohort/src/` (Constants.cs in `utility/`).
- `[PlugIn.cs:N]`: `/workspace/LandisII/ref_pnet/Extension-PnET-Succession/src/PlugIn.cs`.
- `[Defaults:N]`: `/workspace/LandisII/ref_pnet/Extension-PnET-Succession/deploy/Defaults/PnETGenericDefaultParameters.txt` (line N).
- `[SRparams]`: `.../deploy/Defaults/SaxtonAndRawlsParameters.txt`.
- `[ex:file]`: example inputs under `.../Extension-PnET-Succession/test/...`.
- `[desc p.N]` = "Description of PnET-Succession v5.1" (PDF page marker N = printed page). `[guide p.N]` = "PnET-Succession v6.0 User Guide" (marker N; printed page = N-2). PDF cites are kept only where they agree with the code.
- "NOT IN SOURCE" = neither code nor PDF gives it. "(judgment)" = my inference or recommendation. "(not read)" = I did not read that code path.

All "Cohort.cs" line numbers refer to `CalculatePhotosynthesis` (lines 903-1557) unless stated. Months are processed as a flat sequence; `Grow()` can be given any number of months (the timestep), 12 per year [SiteCohorts.cs:1378, PlugIn.cs:579-587].

---------------------------------------------------------------------

## 1. Model overview

### 1.1 Time step and structure
- Internal step is **monthly**; the succession "Timestep" only sets how many months of climate are fed to one `Grow()` call [PlugIn.cs:579-587; desc p.5]. With Timestep = 1 each `Grow()` gets 12 months of one year.
- Monthly climate record per ecoregion/site: Tmin, Tmax, Prec, PAR (PAR0), CO2, optional O3 [EPV.cs:306-330; ObservedClimate.cs]. Daylength etc. are derived per month (Section 3).
- Dates used are the **15th of each month** (`new DateTime(year,1,15)` then AddMonths(1)), so day-of-year for daylength is the mid-month day [PlugIn.cs:212, 579; EPV.cs:317].
- Competition is for light and water only. N enters only through the fixed FolN.
- Cohort = species-age. Cohort pools are densities (g/m2) within the cohort's own canopy "cube"; site-level values are weighted by `CanopyLayerProp` (Section 4).

### 1.2 What a cohort carries (pools)
Persistent live pools [CohortData.cs:27-61; Cohort.cs:106-171]:

| Field | Meaning | Units |
|---|---|---|
| `TotalBiomass` | structural biomass **wood + root combined** (single pool) | gDW/m2 |
| `Fol` | foliage | gDW/m2 |
| `NSC` | non-structural carbon (reserve) | **gC**/m2 |

Derived (not stored independently):
- `Wood = round((1-FracBelowG)*TotalBiomass)`, `Root = round(FracBelowG*TotalBiomass)` [Cohort.cs:156-171].
- `AGBiomass = (1-FracBelowG)*TotalBiomass + Fol` [Cohort.cs:707, CohortData.cs:27].
- **LANDIS-reported `Biomass` = `(int)(AGBiomass * CanopyLayerProp)`** = (aboveground wood + foliage) scaled to the site [Cohort.cs:120-126]. Roots are never reported. (The guide agrees: wood+foliage, [guide p.20]; the canopy weighting is a code detail.)
- `FActiveBiom = exp(-FrActWd * BiomassMax)`, where BiomassMax is the **lifetime max of TotalBiomass** [Cohort.cs:514-520]. It is an activity multiplier on wood, not a pool.
- `NSCfrac = NSC / (FActiveBiom * (TotalBiomass + Fol) * CFracBiomass)`; `IsAlive = NSCfrac > 0.01` [Cohort.cs:219-225, 524-530].
- `Fage = max(0, 1 - (Age/Longevity)^PsnAgeRed)` [Cohort.cs:210-216].

Two cell-level dead pools (woody debris, leaf litter) exist at site level; litter decay constant `KNwdLitter = max(0.3, (-0.5365 + 0.00241*AETsum) - (-0.01586 + 0.000056*AETsum)*FolLignin*100)` with AETsum = sum of the 12 monthly AET values [SiteCohorts.cs:4559-4566]. Woody debris decays with KWdLit (Pool class not read).

### 1.3 Canopy structure
- Each cohort = `IMAX` equal-foliage-mass **sublayers** (default 5) [Defaults:6; Globals.IMAX]. Each sublayer is processed separately each month; sublayer `index` 0 = top.
- Cohorts are assigned to **major canopy layers** by `BiomassMax` (Section 4.1); layers are processed top (highest index) to bottom.
- Within a layer cohorts occupy side-by-side non-overlapping shares of the cell: `CanopyLayerProp` (Section 4.2).
- Layer exit PAR = weighted/product combination of cohort transmissivities (Section 4.3); the exit PAR of the lowest layer is the subcanopy PAR.

### 1.4 Order of operations (summary; full trace in Section 7)
Per month, site level: climate derivations -> snow -> (soil ice, off by default) -> new cohort sublayers allocated -> interception, PrecLossFrac, precipitation events assigned randomly to sublayers -> for each layer (top to bottom), for each sublayer of each cohort in random order: **fast water inputs, maintenance respiration, (December: mortality check, turnover, NSC->structure allocation, age+1), phenology/foliage allocation, fRad, fWater, photosynthesis, transpiration, foliar respiration, NSC update**, then ground evaporation for this sublayer's events -> layer PAR transmission and canopy proportions -> site monthly bookkeeping. After the last month: establishment draw and removal of cohorts with NSCfrac <= 0.01.

### 1.5 What "biomass" means to the rest of LANDIS
- Reported cohort biomass = `(wood_AG + foliage) * CanopyLayerProp` [Cohort.cs:120-126, 710, 1069].
- For our FIA mapping (judgment): FIA aboveground biomass per plot-cohort corresponds to this quantity, but it is **canopy-proportion-weighted**; the solve in Section 8 inverts it.

---------------------------------------------------------------------

## 2. Parameters

Defaults come from `PnETGenericDefaultParameters.txt` (a parameter may live in the generic file, species file or ecoregion file but not in more than one) [Defaults:1-76; guide p.40]. Real species values: `[ex: One_Cell_Real/GenericPnETSpeciesParameters.txt]`.

### 2a. Species parameters (PnETSpeciesParameters; `ISpeciesPnET`)

| Name | Meaning | Units | Default / typical | Cite |
|---|---|---|---|---|
| FolN | foliar N | % by weight | no default; 0.75-1.93 in ex (decid ~1.4-1.9; conifer 0.75) | [ex: GenericPnETSpecies...]; desc p.33 |
| AmaxA, AmaxB | intercept, slope of Amax = AmaxA + AmaxB*FolN | nmol CO2/g Fol/s | evergreen 5.3, 21.5; deciduous -46, 71.9 | [EPV.cs:348-350; Cohort.cs:1401]; guide p.83 |
| AmaxFrac | mean daily / early-morning Amax | - | 0.75 | [Defaults:59] |
| CO2AMaxBEff | relative AmaxB change at 550 vs 350 ppm | - | 1.0 (off) | [Defaults:37] |
| BFolResp | base foliar resp / Amax (used only if Wythers=false) | - | 0.1 | [Defaults:10; EPV.cs:393] |
| Q10 | respiration Q10 (only if Wythers=false) | - | 2 | [Defaults:13; EPV.cs:394] |
| MaintResp | maintenance respiration rate | per month, of (TotalBiomass*CFracBiomass) | 0.002 | [Defaults:18; Cohort.cs:1028] |
| HalfSat | PAR at fRad = 0.5 | umol/m2/s (or W/m2) | 99-250 in ex | [ex]; guide p.47 |
| CO2HalfSatEff | slope of HalfSat vs CO2 | - | 0 (off) | [Defaults:40; Cohort.cs:1229-1230] |
| k | light extinction coefficient | - | 0.5 evergreen, 0.58 deciduous | [ex]; guide p.83 |
| SLWmax | specific leaf weight at top | g/m2 | 47-170 in ex | [ex] |
| SLWDel | SLW decrease with cumulative foliage above | m2/g (per g Fol above) | 0 evergreen, 0.2 deciduous | [Defaults:33; Cohort.cs:1864-1904] |
| MaxLAI | cap used for canopy proportion | m2/m2 | if -9999: computed (Section 4.2) | [SpeciesPnET.cs:232-256; Defaults:70] |
| FracFol | foliage per unit active structural biomass | - | 0.033 default; 0.019 decid, 0.06 fir in ex | [Defaults:22] |
| MaxFracFol, FracFolShape | open-light foliage multiplier and shape | -, - | MaxFracFol = FracFol if -9999; shape 0; ex shape 5-6 | [Defaults:44-45; SpeciesPnET.cs:217-221] |
| FrActWd | active-xylem decay: FActiveBiom = exp(-FrActWd*BiomassMax) | per g/m2 | 0.00004 | [Defaults:21] |
| MaxFolN, FolNShape | dynamic FolN | %N, - | MaxFolN = FolN if -9999 (off); shape 0 | [Defaults:41-42; SpeciesPnET.cs:212-216] |
| TOfol | foliage turnover (fraction lost at leaf-off) | per year | 1 deciduous, 0.25-0.5 evergreen | [ex]; guide p.47 |
| TOroot, TOwood | root, wood turnover | per year | 0.02, 0.01 (default); 0.015-0.025 wood in ex | [Defaults:11-12] |
| FracBelowG | root share of TotalBiomass | - | 0.33 | [Defaults:23] |
| DNSC | target NSC fraction (reserve) | - | 0.05 | [Defaults:17] |
| CFracBiomass | C content of dry weight | gC/gDW | 0.45 | [Defaults:19] |
| InitialNSC | NSC of newly established cohort | gC/m2 | 7 | [Defaults:16] |
| PsnTMin, PsnTOpt, PsnTMax | Tday limits for photosynthesis | C | PsnTMax = 2*Opt - Min if -9999 | [Defaults:52]; guide p.48 |
| PsnAgeRed | exponent in Fage | - | 5 | [Defaults:7] |
| LeafOnMinT | Tmin threshold for leaf-on/off | C | = PsnTMin if -9999 | [Defaults:51; SpeciesPnET.cs:223-226] |
| ColdTol | cold-kill temperature | C | -9999 (off) | [Defaults:46] |
| H1, H2, H3, H4 | Feddes pressure heads (positive m) | m | 0, 3.64, 150, 150 | [Defaults:24-27] |
| DVPD1, DVPD2 | fDVPD = max(0, 1 - DVPD1*VPD^DVPD2) | kPa-based | **0.05, 2** | [Defaults:8-9; EPV.cs:338] |
| NSCReserve | NSC retained in lower layers before foliage allocation | - | 0 | [Defaults:60; Cohort.cs:1141] |
| Longevity | from LANDIS species file | years | - | [Cohort.cs:214] |
| KWdLit, FolLignin | litter decay | per yr, g/g | 0.1, 0.2 | [Defaults:14-15] |
| LifeForm | "ground" in name -> restricted to lowest layer | string | tree | [Defaults:61; SiteCohorts.cs:1431-1434] |
| EstRad, EstMoist, MaxPest, PreventEstablishment | establishment | - | 1, 1, 1.0, false | [Defaults:20, 28-29, 53] |
| O3StomataSens, O3GrowthSens | ozone | category, - | NA, 0 | [Defaults:38-39] |
| RefolMinimumTrigger/Maximum/Cost/NonRefolCost | refoliation after defoliation (disturbance only) | - | 0.60 / 0.70 / 0.90 / 0.10 | [Defaults:66-69] |
| InitBiomass | derived: `(int)(311*...)`, see formula | gDW/m2 | ~306 with defaults | [SpeciesPnET.cs:410-411] |

Derived `InitBiomass`: `ib = (uint)(InitialNSC/(DNSC*CFracBiomass))` (=311 with defaults); `InitBiomass = (int)( ib - (uint)(FracBelowG*ib)*TOroot - (uint)((1-FracBelowG)*ib)*TOwood )` [SpeciesPnET.cs:410-411, 423-424]. This is the death threshold of Section 6.5.

### 2b. Ecoregion / site parameters [EcoregionData.cs:~366-380; Hydrology.cs:139-171]

| Name | Meaning | Units | Default | Cite |
|---|---|---|---|---|
| SoilType | key into the Saxton-Rawls table | string | - | [SRparams] |
| Latitude | for daylength | deg | - | [EPV.cs:306] |
| RootingDepth | soil bucket depth | mm | - (1000 in ex) | [ex] |
| PrecLossFrac | fraction of throughfall lost to surface runoff | - | - | [SiteCohorts.cs:1932] |
| LeakageFrac | fast leakage fraction of water above FC | - | 1 | [Defaults:31] |
| PrecIntConst | interception coefficient | - | 0.1 | [Defaults:32] |
| PrecipEvents | events per month | count | 11 | [Defaults:35] |
| SnowSublimFrac | fraction of new snow lost before it joins the pack | - | 0.15 | [Defaults:43; SiteCohorts.cs:1615] |
| EvapDepth | soil depth for ground evaporation | mm | **500** in Defaults (guide prints "005"; the ex file uses 25) | [Defaults:62; Hydrology.cs:251] |
| RunOffCapture | standing-water depth above saturation | mm | 0 | [Defaults:50] |
| WinterSTD | SD for cold-kill | C | 0 | [Defaults:47] |
| SoilIceDepth, LeakageFrostDepth, FrostFactor, MossDepth | soil ice (off) | - | false, 0, 1, 0 | [Defaults:48-49, 54, 65] |
| Derived FieldCap = theta(33 kPa); WiltPnt = theta(1500 kPa); Porosity | volumetric (m3/m3, called "mm/m" in the code) | | | [Hydrology.cs:159, 164, 167] |

### 2c. Generic / global parameters [Defaults]

| Name | Meaning | Default | Cite |
|---|---|---|---|
| MaxCanopyLayers | cap on layers | 3 | [Defaults:4] |
| LayerThreshRatio | layer split ratio | 0.5 | [Defaults:5] |
| IMAX | sublayers per cohort | 5 | [Defaults:6] |
| Wythers | dynamic Q10 + base resp | true | [Defaults:34] |
| DTemp | two-sided temperature response | true | [Defaults:36] |
| ETExtCoeff | ground ET extinction | 0.65 | [Defaults:74] |
| RETCropCoeff | scaler on Hamon RET | 0.3 | [Defaults:75] |
| CanopySumScale | rate of rescaling when layer proportion > 1 | 0.206 | [Defaults:76] |
| CohortStacking | pre-v5 method | false | [Defaults:72] |
| PrecipEventsWithReplacement | sample sublayers with replacement | true | [Defaults:58] |
| PARunits | umol/m2/s or W/m2 | umol | [Defaults:56] |
| InitialCommunitiesSpinup | Spinup / NoSpinup / SpinupLayers / SpinUpLayersRescale | SpinupLayers | [Defaults:64] |
| SpinUpWaterStress | apply H1/H2 during spinup | false | [Defaults:57] |
| MinFolRatioFactor | init sanity check | 0 | [Defaults:71] |

Soil table [SRparams] (sand, clay as fractions, pctOM in %, densFactor, gravel): SAND .85/.04/2.08; LOSA .80/.05/2.33; SALO .63/.10/2.52; LOAM .41/.19/3.06; SILO .15/.18/3.05; SILT .05/.10/1.89; SNCL .53/.26/2.29; CLLO .29/.32/3.51; SLCL .09/.32/4.06; SACL .50/.40/2.50; SICL .08/.47/3.59; CLAY .13/.55/2.88; BEDR .10/.10/1.0 with gravel = 1; densFactor = 1 everywhere, gravel 0 except BEDR.

Constants [Constants.cs:6-9]: MC = 12, MCO2 = 44, SecondsPerHour = 3600, billion = 1e9.

---------------------------------------------------------------------

## 3. Monthly climate-derived variables (per ecoregion-month, per species where noted) [EPV.cs]

Inputs: Tmin, Tmax, Prec (mm), PAR0, CO2 (ppm), O3 (cumulative).
Note: PAR0 is parsed into a `ushort` (integer truncation) in `ObservedClimate` [ObservedClimate.cs:252] (climate-file path; the climate-library path is `double`, [MonthlyClimateRecord.cs:17]).

```
Tave     = 0.5*(Tmin + Tmax)                         [EPV.cs:313]
Tday     = 0.5*(Tmax + Tave) = (3 Tmax + Tmin)/4     [EPV.cs:321]   (DISAGREE with desc p.15; matches guide p.63)
DaySpan  = days in month (Feb = 28)                  [EPV.cs:149-163]
hr       = hours of daylight (below), at the 15th    [EPV.cs:255-292, 317]
Daylength = 3600*hr  (SECONDS)                       [EPV.cs:249-253]
Nightlength = 3600*(24-hr)                           [EPV.cs:243-247]
```
Daylength (hours) [EPV.cs:255-292]:
```
LatRad = lat*2*pi/360
z     = 0.39785*sin(4.868961 + 0.017203*DOY + 0.033446*sin(6.224111 + 0.017202*DOY))
decl  = atan(z/sqrt(1-z^2))            if |z|<0.7 else pi/2 - atan(sqrt(1-z^2)/z)
z2    = -tan(decl)*tan(LatRad)
h     = 0 if z2>=1;  pi if z2<=-1;  else (TA=|z2|; AC = pi/2 - atan(TA/sqrt(1-TA^2)) if TA<0.7 else atan(sqrt(1-TA^2)/TA); h = pi-AC if z2<0 else AC)
hr    = 2*(h*24)/(2*pi) = 24*h/pi
```
(Also computes `r = 1 - 0.0167 cos(0.0172 (DOY-3))`, unused.)

VPD (kPa) [EPV.cs:166-196]:
```
VP(a,b,c,T) = a*exp(b*T/(T+c))
es    = VP(0.61078, 17.26939, 237.3, Tday);   if Tday<0: VP(0.61078, 21.87456, 265.5, Tday)
emean = VP(0.61078, 17.26939, 237.3, Tmin);   if Tmin<0: VP(0.61078, 21.87456, 265.5, Tmin)
VPD   = es - emean
```
Per species [EPV.cs:332-416]:
```
DVPD      = max(0, 1 - DVPD1 * VPD^DVPD2)                          [338]
JH2O      = 0.239 * VPD / (8314.47 * (Tmin + 273))                 [342]
AmaxB_slope = (CO2AMaxBEff-1)*AmaxB/200
AmaxB_int   = -1*((CO2AMaxBEff-1)*1.75 - 1)*AmaxB
AmaxB_CO2   = AmaxB_slope*CO2 + AmaxB_int                          [348-350]
FTempPSN  = DTempResponse(Tday,PsnTOpt,PsnTMin,PsnTMax)  if DTemp  [354-362]
            else CurvelinearPsnTempResponse(...)
if Wythers: BaseFolRespFrac = 0.138071 - 0.0024519*Tave            [382]
            Tmid = 0.5*(Tave + PsnTOpt);  Q10base = 3.22 - 0.046*Tmid  [385-388]
else:       BaseFolRespFrac = BFolResp;  Q10base = Q10             [393-394]
Q10Factor(T) = Q10base^((T - PsnTOpt)/10)                          [418-423]
Q10Factor_foliar = Q10Factor(Tave)                                 [399]
fTempRespDay = Q10Factor(Tday), fTempRespNight = Q10Factor(Tmin)
FTempRespWeightedDayAndNight = min(1, (fTempRespDay*Daylength + fTempRespNight*Nightlength)/(Daylength+Nightlength))   [401-408]
MaintRespFTempResp = MaintResp * FTempRespWeightedDayAndNight       [411]
```
Temperature response of photosynthesis [EPV.cs:199-241]:
- `DTemp=true`: `DTempResponse`: 0 if Tday<PsnTMin or Tday>PsnTMax. If Tday <= PsnTOpt: `max(0, (Tmx_est - Tday)*(Tday - PsnTMin) / ((Tmx_est-PsnTMin)/2)^2)` with `Tmx_est = PsnTOpt + (PsnTOpt - PsnTMin)`. Else: `max(0, (PsnTMax - Tday)*(Tday - Tmn_est) / ((PsnTMax - Tmn_est)/2)^2)` with `Tmn_est = PsnTOpt + (PsnTOpt - PsnTMax)`. Peak = 1 exactly at PsnTOpt (two half-parabolas). DISAGREE with the single parabola in desc p.15 / guide.
- `DTemp=false`: 0 below PsnTMin, 1 above PsnTOpt, `(PsnTMax - T)(T - PsnTMin)/((PsnTMax-PsnTMin)/2)^2` between.

Notes: the formulas use Tave for respiration and Tmin for the gas volume; "Tday" only drives photosynthesis temperature response, VPD and Q10 day factor.

---------------------------------------------------------------------

## 4. Canopy structure equations

### 4.1 Layer assignment [SiteCohorts.cs:4084-4188, 1424-1476]
Input list = `BiomassMax` (max lifetime TotalBiomass, g/m2) of all cohorts. Algorithm (`GetBinsByCohort`):
```
sort BiomassMax descending
layer being built L = {first (largest)}
for each cohort b in order:
    ref   = first (largest) BiomassMax in current layer L
    ratio = b / ref
    if ratio < LayerThreshRatio and (number of layers so far) < MaxCanopyLayers:
        start a new (lower) layer; ref becomes this cohort
    add b to current layer
```
- Layer index: the **top (largest-biomass) layer has the highest index**, lowest layer = 0 [4112-4123]. Once MaxCanopyLayers is reached the remaining small cohorts all fall into the lowest layer.
- Cohorts with identical BiomassMax share a bin; "ground" LifeForm -> layer 0 [SiteCohorts.cs:1431-1434].
- `CohortStacking=true` sets one layer per cohort and ratio threshold 1.
- This resolves the PDF gap on the reference biomass (it is the largest cohort of the layer under construction).

### 4.2 Canopy proportion (CanopyLayerProp) [SiteCohorts.cs:2129-2214]
```
MaxLAI (species, if not given) [SpeciesPnET.cs:236-246]:
   peakBiomass = 1/FrActWd
   peakFoliage = peakBiomass * FracFol * exp(-FrActWd*peakBiomass) = FracFol/(FrActWd*e)
   MaxLAI = sum_{i=0}^{IMAX-1} max(0.01, (peakFoliage/IMAX) / (SLWmax - SLWDel*i*(peakFoliage/IMAX)))

after each layer is processed (all of the layer's sublayers done):
  for cohort c in layer b:
     LAISum = sum(c.LAI)   (this month)
     if c.Leaf_On and LAISum > c.LastLAI:  c.LastLAI = LAISum          [2136-2140]
     c.CanopyLayerProp = min(c.LastLAI/MaxLAI, c.CanopyGrowingSpace)    [2178]
  layerSumCanopyProp[b] = sum_c min(LastLAI/MaxLAI, CanopyGrowingSpace) [2146-2157, 2161]
  if layerSumCanopyProp[b] > 1 and c.growMonth == 1:                    [2180-2189]
       adj = c.CanopyLayerProp / layerSumCanopyProp[b]
       c.CanopyLayerProp = (adj - c.CanopyLayerProp)*CanopySumScale + c.CanopyLayerProp
       c.CanopyGrowingSpace = min(c.CanopyGrowingSpace, c.CanopyLayerProp)  (permanent cap)
  CohortStacking: CanopyLayerProp = CanopyGrowingSpace = 1
```
- `LastLAI` is reset to 0 each year at the cohort's first growing-season month [Cohort.cs:1158] and ratchets up to the season maximum.
- New cohorts: `CanopyLayerProp = LastLAI/MaxLAI` with LastLAI from ideal foliage [Cohort.cs:733-743].
- Matches desc p.17-18 (PCanArea), including `PeakFoliage = (1/FrActWd)*FracFol*e^-1`.

### 4.3 PAR transmission through a layer [SiteCohorts.cs:2141-2149, 2162-2274]
Let `T_c = exp(-k_c * LAISum_c)` and `p_c = min(LastLAI/MaxLAI, CanopyGrowingSpace)` (CanopyLayerProp when the layer is crowded).
```
sumProp = sum_c p_c         (layer)
if sumProp <= 1:   PAR_below = PAR_above * ( sum_c T_c*p_c + (1 - sumProp) )     # open fraction transmits 1
else:              PAR_below = PAR_above * prod_c T_c^(CanopyLayerProp_c)         # product of powers
if the layer is empty (no AGBiomass): PAR_below = PAR_above
```
`PAR_above` for the top layer = `PAR0` of the month; for lower layers = PAR_below of the layer above. This is the "subcanopy PAR" at the end. (Snow does not zero PAR in the code: the line is commented out [SiteCohorts.cs:2555-2559]; guide p.17 says otherwise, DISAGREE.)

### 4.4 LAI and SLW [Cohort.cs:1864-1904]
```
LAI_i = min( LAImax_i , (1/IMAX)*Fol / ( SLWmax - SLWDel * i * (1/IMAX) * Fol ) ),   i = 0..IMAX-1
LAImax_i = max(0.01, 25 - sum_{j<i} LAI_j)        # cohort LAI capped at 25
if Fol>0 and LAI_i<=0:  LAI_i = LAImax_i/(IMAX - i)  (warning; low SLWmax)
```
So SLW decreases with the cumulative foliage mass above the sublayer (i/IMAX*Fol) times SLWDel. This resolves the garbled PDF form [desc p.10]. Cohort LAI = sum_i LAI_i. The cap of 25 is not in the PDFs.

---------------------------------------------------------------------

## 5. Per-sublayer physiology (as coded; `CalculatePhotosynthesis`, line numbers in Cohort.cs)

Each call processes the cohort's next sublayer `index` (index++ at end, [1555]). A cohort is called IMAX times per month, interleaved with other cohorts of its layer.

### 5.1 Water inputs for this sublayer [917-1014]
- `MeltIn` (this sublayer's share of snowmelt = snowmelt/NSublayers) and `precipCount` events of size `PrecInByCanopyLayer = PrecInByEvent` (if this sublayer was assigned events).
- For melt (if > 0): `waterCapacity = Porosity*RootingDepth*frostFreeProp`; `runoff = min(Melt, max(Water*RootingDepth*ffp + Melt - waterCapacity, 0))`; if RunoffCapture>0 part of runoff goes to SurfaceWater up to RunoffCapture; `Water += (Melt - runoff)/(RootingDepth*ffp)`.
- Per event (p = 1..precipCount): same runoff test with `PrecInByEvent`; add to Water; then **fast leakage** `leak = max(LeakageFrac*(Water - FieldCap),0)*RootingDepth*ffp` (mm), subtract; surface water drains into soil up to Porosity.
- `Water` is volumetric (m3/m3); `AddWater(mm, depth)` does `Water += mm/depth`, floored at 0 [Hydrology.cs:82-99]. With soil ice off, ffp = frostFreeProp = 1.

### 5.2 Maintenance respiration [1028-1032]
```
MaintRespSub = (1/IMAX) * min( NSC, MaintRespFTempResp * TotalBiomass * CFracBiomass )   [gC]
NSC = max(0, NSC - MaintRespSub)
```
Base is **TotalBiomass (wood+root)** x CFracBiomass, not "active wood" and not including foliage (DISAGREE with guide p.42). MaintRespFTempResp from Section 3. It is charged every month, also in leaf-off months.

### 5.3 December events (only when index == IMAX-1 and Month == 12) [1036-1078]
In this order:
1. Mortality (if AllowMortality): if `!IsAlive` (NSCfrac <= 0.01): `NSC = 0`. Else if CurrentTime>0 and `TotalBiomass < InitBiomass`: `NSC = 0; Leaf_On=false; foliage senescence -> litter; LastFoliageSenescence`.
2. Turnover: `Senescence()`: `sen = Root*TOroot + Wood*TOwood` (rounded Root/Wood); `TotalBiomass -= sen`; this whole amount goes to **woody debris** (roots included, DISAGREE with guide p.80 "root vaporizes") [1058-1060; Cohort.cs:1826-1834]. LastWoodySenescence = sen.
3. NSC -> structure: `Alloc = max(NSC - DNSC*FActiveBiom*TotalBiomass*CFracBiomass, 0)`; `TotalBiomass += Alloc/CFracBiomass`; `BiomassMax = max(BiomassMax, TotalBiomass)`; `NSC -= Alloc` (floor 0). [1066-1073]
4. `Age += 1` [1074].
Note: the December photosynthesis of this last sublayer then adds NetPsn to the already-reduced NSC (carried into next year).

### 5.4 Phenology and foliage allocation (index == 0 only, once per cohort-month) [1081-1223]
```
if coldKillBoolean:  ColdKill = floor(Tave - 3*WinterSTD); Leaf_On=false; NSC=0; foliage -> litter
else if Tmin <= LeafOnMinT:
        if Leaf_On: Leaf_On=false; Fol -= TOfol*Fol -> litter (FoliageSenescence; if NSCfrac<=0.01 all Fol lost)
        growMonth = -1
     else if frostFreeProp>0:
        if !Leaf_On: growMonth = 1 else growMonth += 1;   Leaf_On = true
if Leaf_On:
   if growMonth == 2:  ReduceFoliage(DeFolProp)     # DeFolProp=0 without disturbance
   else:
      if firstYear: adjFracFol = MaxFracFol
      IdealFol = adjFracFol * FActiveBiom * TotalBiomass
      NSClimit = NSC   (if cohort is below the top layer: NSC - NSCReserve*FActiveBiom*(TotalBiomass+Fol)*CFracBiomass)
      if growMonth < 2:                       # i.e. first leaf-on month (growMonth==1)
          if IdealFol > Fol:
              FolCost = max(0, min(NSClimit, CFracBiomass*(IdealFol - Fol)))   [gC]
              FolTentative = FolCost/CFracBiomass                              [gDW]
          LastLAI = 0
      elif growMonth == 3: refoliation (only if DeFolProp > 0: disturbance, skip)
      if FolTentative > 0.01:  Fol += FolTentative;  MaxFolYear = max(MaxFolYear, Fol)
      NSC = max(0, NSC - FolCost)
```
- Foliage is only added in the first leaf-on month, limited by NSC (deficit not allowed here). The `tentativeCanopyProp` computation [1199-1204] has no effect (actualFol = FolTentative).
- `adjFracFol` is updated each December and each January from last season's mean fRad: `adjFracFol = FracFol + (MaxFracFol - FracFol)*meanFRad^FracFolShape`, where meanFRad = mean over leaf-on months of the cohort-mean sublayer fRad; if no record `adjFracFol = FracFol` [Cohort.cs:640-654, 620-628; SiteCohorts.cs:1596, 2615]. When MaxFracFol = FracFol (default) it is a constant.

### 5.5 Light [1226-1233]
```
LAI[index] = CalculateLAI(Fol, index)
AdjHalfSat = CO2HalfSatEff*CO2 + (HalfSat - 350*CO2HalfSatEff)
LayerPAR   = mainLayerPAR * exp(-k * (sum(LAI) - LAI[index]))      # cumulative LAI of sublayers already processed this month
fRad       = 1 - exp(-LayerPAR * ln(2) / AdjHalfSat)
```
`mainLayerPAR` = PAR at the top of the cohort's major layer [SiteCohorts.cs:2047]. The LAI array is re-zeroed each month (`InitializeSubLayers`, [SiteCohorts.cs:1913]) so sum(LAI)-LAI[index] is the LAI above. This agrees with desc p.14 (fRad with the minus sign; the guide extraction lost it).

### 5.6 Water stress [1240-1284]
```
PressureHead(m) = CalculateWaterPressure(Water, SoilType)             (Section 5.9)
fWater = ComputeFWater(H1,H2,H3,H4, PressureHead)    (only when simulation year>0; in spinup H1=H2=-1 unless SpinUpWaterStress)
if frostFreeProp <= 0:  fWater = 0
```
```
ComputeFWater(H1,H2,H3,H4,PH) [1571-1587]:
   if PH <= H1: 0
   elif PH < min(H1,H2) or PH >= H4: 0
   elif PH > H3: 1 - (PH - H3)/(H4 - H3)
   elif PH < H2: PH/(H2-H1) - H1/(H2-H1)
   else 1
```
(fWaterOzone uses H1=H2=-1 and is only for ozone.)

### 5.7 Dynamic FolN and Amax [1286-1294, 1328-1404]
```
adjFolN = FolN + (MaxFolN - FolN) * fRad^FolNShape          (= FolN by default)      [1292]
ciModifier = 1                       (ozone off)             [1298]
if Leaf_On:
   cicaRatio  = -0.075*adjFolN + 0.875
   ciElev     = CO2 * cicaRatio * ciModifier
   Ca_Ci      = CO2 - ciElev
   Gamma = 40;  Ca0 = 350
   DelAmax    = max(0, (ciElev - Gamma)/(ciElev + 2*Gamma) * (Ca0 + 2*Gamma)/(Ca0 - Gamma))     [1373-1392]
   V          = 8314.47*(Tmin + 273)/101.3                                                  [1396]
   JCO2       = 0.139 * ((CO2 - ciElev)/V) * 1e-6                                           [1397]
   JH2O       = JH2O_month / ciModifier                (JH2O_month from Section 3)         [1398]
   Amax       = DelAmax * (AmaxA + AmaxB_CO2 * adjFolN)             [nmol CO2/g Fol/s]     [1401]
   BaseFolResp= BaseFolRespFrac * Amax
   AmaxAdj    = Amax * AmaxFrac
   GrossAmax  = AmaxAdj + BaseFolResp
   RefGrossPsn = DaySpan * (GrossAmax * DVPD * Daylength * MC) / 1e9         [gC/gFol/month]   [1407]
   GrossPsnPotential = (1/IMAX) * FTempPSN * fRad * Fage * RefGrossPsn * Fol                 [gC/m2]  [1413]
```
- `Daylength` is in seconds [EPV.cs:249-253]: confirms the dimensional reading in the earlier draft.
- **fWater is not in GrossPsnPotential**; it is applied in 5.8.
- The PDF forms for Amax, AmaxB_CO2, DelAmax, cicaRatio agree with the code except: the PDF `Ci = Ca*cicaRatio` (no ciModifier) vs code `ciElev = CO2*cicaRatio*ciModifier` (identical when ozone is off).

### 5.8 Transpiration, water limitation, respiration, NetPsn [1416-1541]
```
PotentialTranspiration = 0.0015 * (GrossPsnPotential/(JCO2/JH2O)) * CanopyLayerProp        [mm]     [1420]
wiltPtWater = CalculateWaterContent(H4*9.804139432 [kPa], SoilType)                                  [1446]
availableWater = (Water - wiltPtWater) * RootingDepth * frostFreeProp                        [mm]     [1447]
if PotentialTranspiration > availableWater:
     Transpiration = max(availableWater, 0)
     GrossPsn = (CanopyLayerProp>0) ? (Transpiration/0.0015)*(JCO2/JH2O)/CanopyLayerProp : 0
     fWater   = (PotentialTranspiration>0) ? Transpiration/PotentialTranspiration : 0
else:
     GrossPsn      = GrossPsnPotential * fWater
     Transpiration = PotentialTranspiration * fWater
Water -= Transpiration/(RootingDepth*frostFreeProp)           [1474]
RefFolResp = BaseFolResp * Q10Factor_foliar * DaySpan * (3600*24) * MC / 1e9                [gC/gFol/month]  [1487]
FolResp    = RefFolResp * Fol / IMAX                                          [1491]   (no fWater factor)
NetPsn_nonO3 = GrossPsn - FolResp
NetPsn = NetPsn_nonO3 * FOzone   (FOzone = 1 with no ozone)
NSC   += NetPsn;   NSC = max(0, NSC)                                                 [1537-1539]
```
- If not Leaf_On: NetPsn = FolResp = GrossPsn = Transpiration = 0 [1542-1552].
- NetPsn is in gC/m2 **of the cohort's own canopy area** (no CanopyLayerProp scaling; only transpiration is scaled by CanopyLayerProp).
- `0.0015` combines 18/12/1000 (mol H2O per mol C, kg per mm); derivation (judgment) and equal to the PDF constant [desc p.9].
- Ozone (off in our first version): `ciModifier = min(1, fWaterOzone + (a*fWaterOzone + b)*o3_cum)` with (a,b) = (-0.021,0.0087) Tolerant, (-0.0148,0.0062) Intermediate, (-0.0176,0.0118) Sensitive; `O3Effect = min(1, lastO3Effect + 0.0026*O3GrowthSens*gwv*o3_month*relO3)`; `relO3 = min(1, 1 - (Layer/nLayers*(0.6163+0.00105*Fol))^3)` [1299-1320, 1520-1533, 1589-1599].

### 5.9 Soil water retention (Saxton and Rawls) [SR.cs:98-175, 63-96]
Per soil type (sand, clay as fractions; pctOM; densFactor; gravel):
```
pm1500   = -0.024*sand + 0.487*clay + 0.006*OM + 0.005*sand*OM - 0.013*clay*OM + 0.068*sand*clay + 0.031
pm1500a  = pm1500 + 0.14*pm1500 - 0.02
pm33     = -0.251*sand + 0.195*clay + 0.011*OM + 0.006*sand*OM - 0.027*clay*OM + 0.452*sand*clay + 0.299
pm33a    = pm33 + (1.283*pm33^2 - 0.374*pm33 - 0.015)
por33    = 0.278*sand + 0.034*clay + 0.022*OM - 0.018*sand*OM - 0.027*clay*OM - 0.584*sand*clay + 0.078
por33a   = por33 + (0.636*por33 - 0.107)
satPor33 = por33a + pm33a;   sandAdjSat = satPor33 + (-0.097*sand + 0.043)
density_OM = (1 - sandAdjSat)*2.65;   density_comp = density_OM*densFactor
Porosity = 1 - density_comp/2.65
moist33_comp = pm33a + 0.2*((1 - density_comp/2.65) - (1 - density_OM/2.65))
B = (ln 1500 - ln 33)/(ln moist33_comp - ln pm1500a)
A = exp( ln 33 + B*ln moist33_comp )
tension(kPa) = A * theta^(-B)           (theta <= Porosity; else 0)
PressureHead(m) = round( tension*0.1019977334 , 2 )
theta(tension) = (tension/A)^(1/(-B))           [CalculateWaterContent]
FieldCap = theta(33 kPa);  WiltPnt = theta(1500 kPa);  Porosity as above        [Hydrology.cs:159-167]
```
OM here is `pctOM` as given in the table (value like 2.08). The units mix (sand/clay fractions, OM in %) are as in the code [SR.cs:127-134]. Initial site Water = FieldCap [SiteCohorts.cs:393]. These are the exact equations behind the PDF's "Saxton and Rawls 2006" statement [guide p.24].

### 5.10 Ground evaporation (after each sublayer, per assigned event) [SiteCohorts.cs:2107-2126; Hydrology.cs:243-271, 379-393, 419-432]
```
RET (mm/day)  = 0 if Tave<0 else 1.2*0.165*216.7 * (Daylength/3600/12) * es/(Tave+273.3),  es = 6.108*exp(17.27*Tave/(Tave+237.3))
groundPET (mm/month) = RETCropCoeff * RET * exp(-ETExtCoeff * sum(CanopyLAI)) * DaySpan          (CanopyLAI = last month's layer-weighted LAI)
groundPETbyEvent = groundPET/PrecipEvents
for each event assigned to this sublayer (if snowPack == 0 and groundPET>0 and ffp>0):
    evapSoilDepth = min(RootingDepth*ffp, EvapDepth)
    evap = max(0, min(groundPETbyEvent, (Water - WiltPnt)*evapSoilDepth))      [mm]
    Water -= evap/(RootingDepth*ffp);  Evaporation += evap
```
(DISAGREE: PDFs describe a single monthly evaporation with `Water*EvapDepth` [desc p.13] or `(Water-WiltPt)*EvapDepth` [guide p.24] and mention "0 when LAI>3"; the code evaporates per precipitation event only, and only for events landing on processed sublayers, with the wilting-point form.)

---------------------------------------------------------------------

## 6. Site-level monthly water and snow (before/after the sublayer loop)

### 6.1 Snow [SiteCohorts.cs:1368-1376, 1611-1621]
```
maxMelt  = 2.74 * max(0,Tave) * DaySpan
snowmelt = min(snowPack, maxMelt)                       # computed BEFORE adding this month's snow
snowFrac = clamp((Tave - 2)/(-7), 0, 1)
newsnow  = snowFrac * Prec
snowPack += newsnow*(1 - SnowSublimFrac) - snowmelt
```
(The desc p.12 versions have typographical errors; code is clear.)

### 6.2 Throughfall and events [SiteCohorts.cs:1920-1979]
```
newrain      = Prec - newsnow                                  (full new snow removed, sublimation not added back)
interception = newrain * (1 - exp(-PrecIntConst * sum(CanopyLAI)))      # CanopyLAI = previous month's layer-weighted LAI; 0 initially
surfaceRain  = newrain - interception
precLoss     = surfaceRain * PrecLossFrac
precin       = surfaceRain - precLoss
PrecInByEvent = precin / PrecipEvents
MeltInWater  = snowmelt
```
Event assignment: draw `PrecipEvents` integers uniformly in 1..N (N = total number of cohort sublayers on the site), with replacement (default) or cycling without replacement. The integer refers to the **position in the processing order** (`subCanopyIndex` counter), not a fixed sublayer identity. Event count per position = `precipCount`. Each assigned position receives `precipCount` events of `PrecInByEvent` [SiteCohorts.cs:1954-2079]. Snowmelt is split equally: `subCanopyMelt = MeltInWater/N` for every sublayer [2066]. `InterceptCumulative += interception/(number of distinct positions)` for each such position [2075] (bookkeeping only, for AET).

If there are no cohorts: melt and events are applied directly to the soil with runoff, fast leakage and evaporation per event [2279-2378].

### 6.3 End-of-month site bookkeeping [2379-2606]
- `CanopyLAI[layer] = layerWtLAI[layer]` (= sum over cohorts of `LAISum_c * min(LastLAI/MaxLAI, CanopyGrowingSpace)`; set only if the layer has AGBiomass) used next month for interception and ground PET [2507-2513, 2270].
- `canopylaimax = max(canopylaimax, sum of layer LAI)`; monthly sums of GrossPsn, NetPsn, FolResp, MaintResp (cohort sums x CanopyLayerProp), Transpiration, Evaporation, Interception, Leakage, Runoff; AET = Evaporation + Transpiration + Interception; PET.
- `StoreFRad()`: for Leaf_On cohorts append the mean sublayer fRad of this month to `LastSeasonFRad` [Cohort.cs:620-628].
- Sublayer arrays nulled (`NullSubLayers`).
- December: litter and woody-debris decomposition; `CalcAdjFracFol()` for every cohort; establishment fWater/fRad accumulation (Section 6.6). January: `ResetFoliageMax`, `LastAGBio = AGBiomass`, `CalcAdjFracFol`, `ClearFRad`.

### 6.4 End of Grow() [SiteCohorts.cs:2662-2715]
Establishment draw (if year>0), then `RemoveMarkedCohorts()` which removes every cohort with `IsAlive == false` (NSCfrac <= 0.01, includes cold-killed and the NSC=0 cohorts from December) [4344-4362]. The mortality classification "cold" vs "succession" only changes accounting.

### 6.5 Mortality summary (code)
1. `NSCfrac <= 0.01` at December (before allocation) -> NSC=0; removed at end of Grow [Cohort.cs:1044-1047].
2. `TotalBiomass < InitBiomass` at December (year>0) -> NSC=0 [1048-1056]. (Implication for FIA-initialised small cohorts: those with TotalBiomass below ~306 g/m2 die at the first December. (judgment))
3. `IsAlive` checked again at the end of Grow.
4. Cold kill: extremeMinTemp = min over the timestep's months of `Tave - 3*WinterSTD`; species whose `ColdTol` exceeds it are killed in that month (`ColdKill=floor(...)`, NSC=0, leaf off) [SiteCohorts.cs:1535-1559, Cohort.cs:1083-1091].
5. No age-based removal at Longevity (fAge -> 0 only reduces photosynthesis; `Fage` is clamped at 0).
6. Canopy-space loss (not mortality): `CanopyLayerProp` rescale.

### 6.6 Establishment (disabled in our first version; brief) [SiteCohorts.cs:2531-2698; EstProb.cs:60-110]
Per month with Tmin>PsnTMin and Tmax<PsnTMax: `fRad_est = min(1, fRad(PAR_sub, AdjHalfSat)^2 / EstRad^2)` and `fWater_est = min(1, fWater(...)^2/EstMoist^2)` (note: the code passes `CalculateWaterContent(Water,...)` into fWater, apparently a bug, [EstProb.cs:84]). Per timestep, mean over the 2nd-4th active months (see code at [2622-2657]); `Pest = mean_fWater * mean_fRad * MaxPest`; establish if `Pest > U(0,1)`. New cohort: age 1, `NSC = InitialNSC`, `TotalBiomass = max(1, InitialNSC/(DNSC*CFracBiomass))` [Cohort.cs:721-753].

---------------------------------------------------------------------

## 7. Order of operations per month (from code)

`SiteCohorts.Grow(data)` [SiteCohorts.cs:1378-2716]. Before the month loop (once per call):
- B1. `establishmentProbability.ResetPerTimeStep()`. Compute `SiteAboveGroundBiomass`; (year>0) per-cohort `CalculateDefoliation` (disturbance only) [1397-1405].
- B2. Build lists `CohortBiomassList = TotalBiomass`, `CohortMaxBiomassList = BiomassMax`. Compute layers via `GetBinsByCohort(CohortMaxBiomassList)` [1424]. Create one entry per (cohort, sublayer) in `SubCanopyCohorts`; assign sublayer ids to layer bins (ground -> layer 0) [1429-1462].
- B3. `random_range[b]` = random permutation of the sublayer ids of each layer [1480; 1320-1348]. This permutation is fixed for the whole `Grow()` call (all months of the timestep).
- B4. Cold-kill pre-check (year>0): `extremeMinTemp`, `coldKillMonth[species]` [1535-1559]. Reset per-timestep accumulators; `lastOzoneEffect[]=0`.

For each month m [1565-2661]:
1. `Ecoregion.Variables = data[m]`; `subcanopypar = PAR0`; reset transp, interception [1567-1571].
2. If Month==1: reset annual accumulators; for every cohort `ResetFoliageMax(); LastAGBio = AGBiomass; CalcAdjFracFol(); ClearFRad()` [1574-1600].
3. Ozone dose increments (skip).
4. **Snow**: snowmelt, newsnow, snowPack update (Section 6.1) [1611-1622].
5. `propRootAboveFrost = 1; leakageFrac = LeakageFrac`; soil temperature block (runs but only affects hydrology if `SoilIceDepth`; skip) [1626-1903].
6. `InitializeSubLayers()` for all cohorts (zero arrays, index=0) [1913].
7. Compute RET (Hamon), `newrain`, `interception` (previous-month canopy LAI), `precLoss`, `precin`, `PrecInByEvent`, `groundPET`, `groundPETbyEvent` [1918-1950].
8. Draw event positions (random, with replacement), group by position [1954-1979].
9. Reset `Runoff, Leakage, Evaporation`, set `PE = groundPET`; `PETcumulative=TransCumulative=InterceptCumulative=0` [1982-1989].
10. If cohorts exist: for each layer b from top to bottom [2043]:
    a. `mainLayerPAR = subcanopypar` (PAR entering the layer) [2047]; layer's `sumCanopyProp` [2052-2058].
    b. for each sublayer r in `random_range[b]` (random order): `subCanopyIndex++`; look up `precipCount` for this position; `subCanopyMelt = Melt/N`; `PETcumulative += RET*DaySpan/N` (PET bookkeeping); call **`CalculatePhotosynthesis`** for cohort c's next sublayer:
       - (i) melt + events water, runoff, fast leakage [5.1];
       - (ii) maintenance respiration [5.2];
       - (iii) if December and last sublayer: mortality, turnover, NSC->structure, age+1 [5.3];
       - (iv) if sublayer 0: phenology and foliage allocation [5.4];
       - (v) LAI, fRad [5.5]; fWater [5.6];
       - (vi) adjFolN, ciModifier [5.7]; if Leaf_On: Amax, GrossPsnPotential, transpiration limited by available water, GrossPsn, Transpiration, remove water, FolResp, NetPsn, NSC += NetPsn [5.7-5.8];
       - (vii) `index++`.
       Then `TransCumulative += Transpiration`; then ground evaporation for each event assigned to this position [5.10].
    c. After all sublayers of the layer: for cohorts in the layer update LastLAI, CanopyLayerProp (and rescale when crowded and growMonth==1), `ANPP`, then layer PAR transmission -> `subcanopypar` for the next layer [2129-2274] (Sections 4.2, 4.3).
11. (No cohorts: apply melt and events to soil directly.) [2279-2378]
12. Monthly bookkeeping, new `CanopyLAI` per layer, albedo (skip), pressure-head average, establishment fWater/fRad (disabled), AET/PET [2379-2553].
13. Surface water -> soil if below porosity [2578-2585].
14. Output (skip). `StoreFRad`, `NullSubLayers` [2604-2606].
15. If December: litter/woody debris decomposition; `CalcAdjFracFol`; establishment month filtering [2609-2658]. `wateravg += Water`.

After the loop: establishment draw (year>0); `RemoveMarkedCohorts` [2664-2715].

Order consequences worth preserving (judgment):
- Water is a shared, sequentially depleted resource: results depend on the processing order (top layer first; random within layer). In a massively parallel Julia version the random permutation per site-year is a design choice; the expected behaviour (top layer priority, random within layer) is what the code guarantees.
- A cohort's sublayer k in a month always follows its sublayer k-1 (because `index` increments in call order and only the cohort's own calls increment it).

---------------------------------------------------------------------

## 8. Initialization from observed cohorts (species, age, biomass)

Modes [SiteCohorts.cs:430-1154; guide p.44-45; desc p.23-24]: `NoSpinup`, `SpinupLayers` (default; runs a spin-up only to estimate max biomass and growing space, then rescales), `SpinUpLayersRescale`, `Spinup` (ignore input biomass; start each cohort at age 1 at its birth year and grow). If solving the initial layout fails the code falls back through the other modes with warnings [988-1129].

**NoSpinup constructor** `Cohort(speciesPnET, age, woodBiomass, ...)` [Cohort.cs:801-845]:
```
TotalBiomass = BiomassMax = woodBiomass / (1 - FracBelowG)      # the input "biomass" is treated as wood
adjFracFol   = MaxFracFol;  ColdKill = int.Max
IdealFol     = MaxFracFol * FActiveBiom * TotalBiomass          # FActiveBiom uses BiomassMax = TotalBiomass here
LastLAI      = sum_i LAI_i(IdealFol)  ;  CanopyLayerProp = LastLAI/MaxLAI   (1 if CohortStacking);  CanopyGrowingSpace = 1
Fol = IdealFol only if Leaf_On (false at construction => Fol = 0 initially)
NSC          = DNSC * FActiveBiom * (TotalBiomass + Fol) * CFracBiomass
LastAGBio    = AGBiomass
```
The `SpinupLayers` constructor (extra args maxBiomass, canopyGrowingSpace, lastSeasonAvgFrad) uses `BiomassMax = max(biomass, maxBiomass)` and `adjFracFol` from `CalcAdjFracFol` [847-890].

**Calibration of initial biomass to the input value (the circular solve)** [SiteCohorts.cs:511-947, final pass 836-947]: for each cohort, given input `targetBiomass` (the cohort's `AGBiomass` at creation = input g/m2), find `TotalBiomass` such that
```
(TotalBiomass*(1-FracBelowG) + IdealFol) * min(LAI(IdealFol)/MaxLAI, CanopyGrowingSpace)  =  targetBiomass     (tolerance < 2 g/m2)
IdealFol = adjFracFol * exp(-FrActWd*max(BiomassMax, TotalBiomass)) * TotalBiomass
```
by stepping `TotalBiomass` by 1/2/10/20/100/200 depending on the error size (loop cap 1000). Then:
```
Fol   = IdealFol * (1 - TOfol)        # winter state: deciduous Fol -> 0; this is the "initial" foliage          [966]
NSC   = DNSC * FActiveBiom * (TotalBiomass + Fol) * CFracBiomass                                          [967]
LastLAI, CanopyLayerProp = min(LAI(Fol-based)/MaxLAI, CanopyGrowingSpace)
```
Layer-crowding: `LayerFoliagePotential`/`MainLayerCanopyProp > 1` are rescaled with CanopySumScale as in Section 4.2 and `CanopyGrowingSpace` is set [596-604, 694-703, 845-854].
**Interpretation**: because the solve's target is `(wood + foliage) * canopy proportion` = the reported `Cohort.Biomass`, the input biomass is *aboveground biomass including foliage, canopy-weighted* in this code. DISAGREE with desc p.24 ("cell-scale wood biomass density"); the NoSpinup constructor treats the same number as pure wood (the difference is small for foliage-poor cohorts). Root = `FracBelowG*TotalBiomass` follows from the single TotalBiomass pool (no separate root initialisation).

Initial soil water = FieldCap; snowPack = 0 (field default); litter and woody-debris pools empty unless supplied.

Spin-up reference [SiteCohorts.cs:1277-1346]: cohorts are added in order of age (oldest first) at their birth year, `Cohort(species, speciesPnET, year, name, 1, stacking)`, and `Grow()` runs between births. Needs climate back to the oldest cohort age.

---------------------------------------------------------------------

## 9. PDF vs code disagreements (code wins)

| # | Topic | PDF | Code | Cite |
|---|---|---|---|---|
| 1 | Tday | desc: (Tmin+Tmax)/2 | (Tmax + Tave)/2 (= guide) | EPV.cs:321; guide p.63 |
| 2 | fTempPsn | single parabola | two half-parabolas peaking at PsnTOpt (DTemp=true) | EPV.cs:218-241 |
| 3 | Structural pools | wood, root, foliage, NSC | wood+root are ONE pool (TotalBiomass); split by FracBelowG | CohortData.cs:32; Cohort.cs:156-171 |
| 4 | Foliage target | FracFol x active wood, "active_wood = exp(-FrActWd*biomass)" | `adjFracFol * exp(-FrActWd*BiomassMax) * TotalBiomass` (BiomassMax = lifetime max TotalBiomass) | Cohort.cs:518, 1135 |
| 5 | Maintenance resp. | proportion of active wood | `MaintRespFTempResp*TotalBiomass*CFracBiomass` per month, 1/IMAX per sublayer | Cohort.cs:1028 |
| 6 | NSCfrac | NSC/(FActive*(wood+root+fol)) | extra CFracBiomass factor in denominator | Cohort.cs:223 |
| 7 | Year-end allocation | not quantified | all NSC above `DNSC*FActiveBiom*TotalBiomass*CFracBiomass` converted to TotalBiomass | Cohort.cs:1066-1073 |
| 8 | Turnover destinations | root "vaporizes" [guide p.80] | wood+root turnover both go to woody debris | Cohort.cs:1058-1060 |
| 9 | Age increment | cohort property | `Age++` inside December processing | Cohort.cs:1074 |
| 10 | Evaporation | one monthly step; Water*EvapDepth or (Water-WiltPt)*EvapDepth; zero if LAI>3 | per precip-event, `(Water-WiltPnt)*min(RootingDepth,EvapDepth)`, RETCropCoeff multiplier, RET=0 if Tave<0 | Hydrology.cs:243-271, 379-432 |
| 11 | Slow leakage at month end | guide p.23 | none; only fast leakage after inputs | Cohort.cs:961; SiteCohorts.cs:2326 |
| 12 | Snow | formula typos | clamp((Tave-2)/-7,0,1); melt min(pack, 2.74*Tave*days); sublimation on new snow | SiteCohorts.cs:1368-1376, 1611-1621 |
| 13 | PAR under snowpack | guide p.17: set to 0 | not implemented (commented out) | SiteCohorts.cs:2555-2559 |
| 14 | DVPD defaults | guide p.41 prints "2, 0.1" | DVPD1 = 0.05, DVPD2 = 2 | Defaults:8-9 |
| 15 | EvapDepth default | guide "005" | 500 | Defaults:62 |
| 16 | LAI cap | none | cohort LAI <= 25 | Cohort.cs:1864-1904 |
| 17 | Q10Factor / Wythers | not given | `Q10base^((T-PsnTOpt)/10)`; Wythers: base resp 0.138071 - 0.0024519*Tave, Q10 = 3.22 - 0.046*mid(Tave, PsnTOpt) | EPV.cs:379-423 |
| 18 | VPD, daylength, V, JH2O | not given | explicit | EPV.cs:166-292, 342; Cohort.cs:1396 |
| 19 | Saxton-Rawls table | not given | full equations + soil table | SR.cs:98-175; SRparams |
| 20 | Reference for layer ratio | not stated | ratio to largest BiomassMax of the layer under construction | SiteCohorts.cs:4113-4118 |
| 21 | Initial biomass meaning | wood biomass density (desc p.24) | solve is on (wood+foliage)*CanopyLayerProp; NoSpinup constructor treats it as wood | SiteCohorts.cs:872; Cohort.cs:807 |
| 22 | fRad PAR basis | guide p.69: top of layer | within-cohort cumulative LAI above sublayer (= desc p.14) | Cohort.cs:1232 |
| 23 | Product-of-powers PAR | desc p.20 | used when layer sum > 1; else weighted average with open gap | SiteCohorts.cs:2162-2268 |
| 24 | Canopy rescale | desc: once per year | applied only in the cohort's first growing month (`growMonth==1`) | SiteCohorts.cs:2180-2189 |
| 25 | PAR0 precision | decimal | `ushort` in climate-file reader | ObservedClimate.cs:252 |

---------------------------------------------------------------------

## 10. State variables

### 10.1 Per-cohort state (CohortData and Cohort fields)
Persistent across months/years (needed in a SoA layout):

| Variable | Meaning | Cite |
|---|---|---|
| species id, `Age` (yr) | `Age` incremented each December | CohortData.cs:22; Cohort.cs:1074 |
| `TotalBiomass` | wood+root, gDW/m2 (cohort density) | CohortData.cs:32 |
| `BiomassMax` | lifetime max TotalBiomass | :46 |
| `Fol` | foliage gDW/m2 | :51 |
| `MaxFolYear` | max Fol this year (output/diagnostics) | :56 |
| `NSC` | gC/m2 | :61 |
| `Leaf_On` (bool) and `growMonth` (int; -1 off, 1 = first leaf-on month) | phenology | CohortData.cs:37; Cohort.cs:594 |
| `CanopyLayerProp`, `CanopyGrowingSpace` | canopy share and its permanent cap | :221, :226 |
| `LastLAI` | seasonal max LAI (reset at first leaf-on month) | :121 |
| `LastSeasonFRad` (list) / `LastFRad`, `adjFracFol` | for dynamic FracFol (can be reduced to a running sum and count) | :86, 81, 91 |
| `firstYear` (bool; true only for freshly established cohorts) | forces adjFracFol = MaxFracFol | Cohort.cs:22, 1131-1132 |
| `ColdKill` (int, int.Max = none) | cold-kill marker | :106 |
| `Layer` (byte) | assigned each month from BiomassMax | :111 |
| `LastAGBio`, `AGBiomass`, `BiomassLayerProp` | diagnostics (ANPP) | :126, 27, 216 |
| `LastWoodySenescence`, `LastFoliageSenescence` | annual litter accounting | :71, 76 |
| `DeFolProp` | defoliation (disturbance only) | :66 |
| `UniversalData.Biomass`, `.ANPP` | LANDIS-facing | :22 |

Per-month scratch (arrays of length IMAX, allocated each month): `LAI, GrossPsn, FolResp, NetPsn, MaintenanceRespiration, Transpiration, PotentialTranspiration, FRad, FWater, Water, PressHead, NumEvents, FOzone, Interception, AdjFolN, AdjFracFol, CiModifier, DelAmax` [CohortData.cs:116-211; Cohort.cs:596-618]; scalars `adjFolN, AdjHalfSat, index` [Cohort.cs:18, CohortData.cs:96, 101]. Only LAI (for LastLAI, CanopyLAI, light), fRad (for adjFracFol), and the monthly flux sums are needed beyond the sublayer loop; the rest can be locals.

### 10.2 Per-site state (SiteCohorts, Hydrology)
Persistent:

| Variable | Meaning | Cite |
|---|---|---|
| `hydrology.Water` | volumetric soil water (m3/m3); init FieldCap | SiteCohorts.cs:393; Hydrology.cs:7-18 |
| `hydrology.SurfaceWater` | standing water above saturation (mm) | Hydrology.cs:70 |
| `hydrology.FrozenDepth`, `FrozenWaterContent` | soil ice (0 when off) | Hydrology.cs:8-9 |
| `snowPack` (mm w.e.) | snow | SiteCohorts.cs:23 |
| `CanopyLAI[layer]` | layer-weighted LAI carried from previous month | :24, 2507-2513 |
| `subcanopypar`, `julysubcanopypar`, `subcanopyparmax` | PAR below canopy | :25-27 |
| `canopylaimax`, `wateravg` | diagnostics | :21-22 |
| `AET[12]`, `PET` | annual ET (AET sum drives litter decay) | SiteCohorts.cs:1350-1358 |
| litter pool, woody-debris pool (mass, decay value) | dead pools | SiteVars.Litter / WoodyDebris; Pool.cs |
| `lastTempBelowSnow`, `soilDiffusivity`, soil-temperature profile | soil ice only | :29, 1648 |
| establishment accumulators (cumulativeFwater/Frad, monthlyCount) | per timestep | :1504-1523 |
| `SiteVars.ExtremeMinTemp`, `PressureHead`, `AnnualPE`, `ClimaticWaterDeficit` | diagnostics/outputs | :1549, 2708, 2379, 2553 |

Per-call scratch: `Runoff, Leakage, Evaporation, PE, PET` (hydrology), `lastOzoneEffect[sublayer]`, `PETcumulative, TransCumulative, InterceptCumulative`, `daysOfWinter` (soil temp), `random_range`, event positions.
Ecoregion constants (read-only): Porosity, FieldCap, WiltPnt, soil table coefficients A, B, RootingDepth, LeakageFrac, etc.

---------------------------------------------------------------------

## 11. Minimum viable implementation (growth + mortality of existing cohorts, no establishment/disturbance)

Needed (exact equations in Sections 3-8):
1. Monthly climate derivations: Tave, Tday, VPD, DVPD, daylength (seconds), DaySpan, JH2O, AmaxB_CO2 (or =AmaxB), FTempPSN, Wythers respiration factors, MaintRespFTempResp.
2. Cohort state: TotalBiomass, BiomassMax, Fol, NSC, Age, Leaf_On/growMonth, LastLAI, CanopyLayerProp, CanopyGrowingSpace, adjFracFol.
3. Site state: Water (init FieldCap), snowPack, CanopyLAI (prior month), subcanopy PAR, Saxton-Rawls tables (A, B, Porosity, FieldCap, WiltPnt per soil type).
4. Layering by BiomassMax (4.1), canopy proportion (4.2), layer PAR (4.3), SLW/LAI (4.4).
5. Sublayer loop in layer order with water bookkeeping (5.1-5.10).
6. December: mortality, turnover (wood+root to debris), NSC->structure, Age+1.
7. End-of-call removal of NSCfrac <= 0.01.

Can be dropped initially: ozone, CO2AMaxBEff/CO2HalfSatEff, MaxFolN/FolNShape, soil ice/permafrost/moss, RunoffCapture/SurfaceWater, albedo, dead-pool decay, ColdTol, refoliation, establishment, litter.

Sensitivity statements in the PDFs (calibration appendix, [guide p.72-82]): MaintResp, AmaxA, AmaxB, k, FracBelowG are the most sensitive generic parameters; calibrate FolN, FracFol, FrActWd, SLWmax (foliage amount, LAI) and TOwood/TOroot (peak height) [guide p.74, 80-82]; early growth is controlled by MaxFracFol/FracFolShape; senescence decline by PsnAgeRed. Code confirms the structural reasons: per-year growth = (NSC surplus above the DNSC reserve)/CFracBiomass minus turnover (Cohort.cs:1066-1073, 1826), foliage = adjFracFol * exp(-FrActWd*BiomassMax) * TotalBiomass (Cohort.cs:1135).

My judgment on what biomass-by-age data can identify (not a source statement):
- Identifiable together as a few combinations: asymptote/peak (TOwood/TOroot vs FolN/AmaxB and FracFol), time to peak (initial slope: FracFol/MaxFracFol, FolN), and decline shape (PsnAgeRed with Longevity).
- Poorly separable: FolN vs AmaxA/AmaxB (both set Amax), k vs SLWmax vs FracFol (all set shading/LAI), TOwood vs TOroot (only their sum matters when FracBelowG is fixed; both just lower TotalBiomass), MaintResp vs TOwood.
- Suggest fixing the generic parameters (AmaxA/B, k, Q10, DVPD1/2, BFolResp via Wythers, DNSC, IMAX, MaxCanopyLayers, LayerThreshRatio) from guide p.83 and fitting per-species/per-group FolN, FracFol (or an overall productivity scalar), FrActWd, SLWmax, TOwood/TOroot, MaxFracFol and PsnAgeRed.

---------------------------------------------------------------------

## 12. Remaining gaps

- Dead-pool `Pool.Decompose()` and AddMass internals (not read); litter is not needed for the first version.
- Soil temperature/permafrost block, ozone, and refoliation details only summarised.
- `Names.cs`/parameter-file parsing defaults beyond `Defaults` (e.g. where PsnTMax = -9999 is converted) not traced; the Defaults comment gives the rule.
- How the climate library (vs climate file) supplies PAR0 precision and CO2/O3 is only touched on.
- Randomness: the sublayer permutation and event positions are stochastic; reproducibility choices are ours.
- Establishment is described only structurally (Section 6.6); the apparent water-content/pressure-head mix-up in `EstablishmentProbability` was not resolved.
