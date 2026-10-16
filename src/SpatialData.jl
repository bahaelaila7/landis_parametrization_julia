# Spatial (raster / TreeMap / LANDIS-input) data loaders for the forward simulations `simulate_spatial_treemap`, `simulate_spatial_landis` and the
# LANDIS export. Independent of the parametrization data layer (`Data`: cohort selection, split, tiering): nothing in a calibration run
# calls it, and `Data` does not include it.
module SpatialData
    using ..PanCore
    include("data/spatial.jl")
end
