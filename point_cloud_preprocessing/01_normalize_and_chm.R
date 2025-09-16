# Load required libraries
library(lidR)
library(raster)
library(terra)
library(sf)
library(jsonlite)

# Ground classification function using DEM (for catalog chunks)
classify_ground_by_dem <- function(las, dem_raster, tol) {

# Convert raster to terra SpatRaster if needed
if (class(dem_raster)[1] == "RasterLayer") {
dem <- terra::rast(dem_raster)
} else {
dem <- dem_raster
}

# Create points for extraction
points_df <- data.frame(
X = las@data$X,
Y = las@data$Y,
Z = las@data$Z
)

# Create SpatVector from points
pts <- terra::vect(points_df[, c("X", "Y")],
geom = c("X", "Y"),
crs = terra::crs(dem))

# Extract DEM values at point locations
dem_values <- terra::extract(dem, pts, ID = FALSE)
dem_col_name <- names(dem_values)[1]

# Check for NA values
na_count <- sum(is.na(dem_values[[dem_col_name]]))

# Calculate vertical distances for valid points
valid_idx <- which(!is.na(dem_values[[dem_col_name]]))
z_diff <- abs(points_df$Z[valid_idx] - dem_values[[dem_col_name]][valid_idx])

# Initialize classification (1 = unclassified)
classification <- rep(1, nrow(points_df))

# Classify ground points (within tolerance)
ground_indices <- valid_idx[z_diff <= tol]
classification[ground_indices] <- 2 # ASPRS ground class

# Classify vegetation points (beyond tolerance)
veg_indices <- valid_idx[z_diff > tol]
classification[veg_indices] <- 5 # ASPRS vegetation class

# Update LAS classification (ensure integer type)
las@data$Classification <- as.integer(classification)

return(las)
}

# Load metadata for z-error threshold calculation
metadata <- fromJSON("data/point_cloud/metadata.json")
z_errors <- sapply(metadata$acquisitions, function(x) x$z_error)
mean_z_error <- mean(z_errors)
cat(sprintf("Ground threshold: %.3f m (mean z-error)\n", mean_z_error))

# File paths for three timepoints
las_files <- c(
"data/point_cloud/1000.las",
"data/point_cloud/1200.las",
"data/point_cloud/1500.las"
)
ellipsoidal_dem <- "data/raster/dems/qspatial_2019_ellipsoidal.tif"
output_chm <- "data/raster/dems/canopy_height.tif"
target_resolution <- 0.05 # meters

# Validate input files
for (las_file in las_files) {
if (!file.exists(las_file)) {
stop(paste("Required file not found:", las_file))
}
}
if (!file.exists(ellipsoidal_dem)) {
stop(paste("Required file not found:", ellipsoidal_dem))
}

# Function to calculate shared bounding box
get_shared_bounds <- function(las_files) {
cat("\nStep 1: Calculating shared bounding box...\n")

bounds_list <- list()
for (i in 1:length(las_files)) {
las <- readLAS(las_files[i], select = "xyz") # Only read coordinates for speed
bounds_list[[i]] <- list(
xmin = min(las@data$X),
xmax = max(las@data$X),
ymin = min(las@data$Y),
ymax = max(las@data$Y)
)
cat(sprintf(" %s: X(%.2f,%.2f) Y(%.2f,%.2f)\n",
basename(las_files[i]),
bounds_list[[i]]$xmin, bounds_list[[i]]$xmax,
bounds_list[[i]]$ymin, bounds_list[[i]]$ymax))
}

# Calculate intersection (minimum bounding box)
shared_bounds <- list(
xmin = max(sapply(bounds_list, function(b) b$xmin)),
xmax = min(sapply(bounds_list, function(b) b$xmax)),
ymin = max(sapply(bounds_list, function(b) b$ymin)),
ymax = min(sapply(bounds_list, function(b) b$ymax))
)

cat(sprintf(" Shared bounds: X(%.2f,%.2f) Y(%.2f,%.2f)\n",
shared_bounds$xmin, shared_bounds$xmax,
shared_bounds$ymin, shared_bounds$ymax))
cat(sprintf(" Shared area: %.1f x %.1f m\n",
shared_bounds$xmax - shared_bounds$xmin,
shared_bounds$ymax - shared_bounds$ymin))

return(shared_bounds)
}

# Calculate shared bounding box
shared_bounds <- get_shared_bounds(las_files)

# Create template raster for consistent CRS and extent
las_crs <- sf::st_crs(readLAS(las_files[1], select = "xyz"))$wkt
template_raster <- terra::rast(xmin = shared_bounds$xmin,
xmax = shared_bounds$xmax,
ymin = shared_bounds$ymin,
ymax = shared_bounds$ymax,
res = target_resolution,
crs = las_crs)

cat(sprintf(" Template raster: %d x %d pixels, CRS: %s\n",
ncol(template_raster), nrow(template_raster),
substr(las_crs, 1, 50)))

cat("\nStep 2: Loading ellipsoidal DEM...\n")
cat(sprintf(" Input: %s\n", ellipsoidal_dem))
dtm <- raster(ellipsoidal_dem)
cat(sprintf(" DEM dimensions: %d x %d pixels\n", ncol(dtm), nrow(dtm)))
cat(sprintf(" DEM resolution: %.3f m\n", res(dtm)[1]))
cat(sprintf(" DEM elevation range: %.2f to %.2f meters\n", cellStats(dtm, min), cellStats(dtm, max)))


# Function to process a single timepoint
process_timepoint <- function(las_file, template, dtm, mean_z_error) {
timepoint <- tools::file_path_sans_ext(basename(las_file))
cat(sprintf("\n Processing timepoint %s...\n", timepoint))

# Read LAS file
las <- readLAS(las_file)
cat(sprintf(" Loaded %s points\n", format(npoints(las), big.mark = ",")))

# Fix return metadata for photogrammetric point cloud
las@data$ReturnNumber <- as.integer(1)
las@data$NumberOfReturns <- as.integer(1)

# Classify ground points
cat(" Classifying ground points using DEM...\n")
las <- classify_ground_by_dem(las, dtm, mean_z_error)

# Normalize heights using TIN from classified ground points
cat(" Normalizing heights using TIN...\n")
las_norm <- normalize_height(las, algorithm = tin())

# Clamp negative values to 0
las_norm@data$Z[las_norm@data$Z < 0] <- 0.0

# --- THIS IS THE CRUCIAL FIX ---
# Clip the point cloud to the template's extent before rasterizing.
# This prevents the "point out of raster" error.
cat(" Clipping point cloud to shared extent...\n")
bbox <- raster::extent(template)
las_clipped <- clip_rectangle(las_norm, bbox@xmin, bbox@ymin, bbox@xmax, bbox@ymax)

# Check if any points remain after clipping
if (is.empty(las_clipped)) {
cat(" Warning: No points remaining after clipping. Skipping this timepoint.\n")
return(NULL)
}

# Generate CHM using the clipped point cloud and the pitfree algorithm
cat(sprintf(" Generating CHM using template grid (%.3f m resolution)...\n", xres(template)))
chm <- rasterize_canopy(
las_clipped, # <-- Use the clipped LAS object
res = template, # Use the template raster directly
algorithm = p2r(subcircle = 0.5 * xres(template), na.fill = knnidw())
)

# Clamp negative values that may be introduced by interpolation
chm[chm < 0] <- 0

# Apply Gaussian smoothing to reduce noise
cat("  Applying Gaussian smoothing...\n")
chm_terra <- terra::rast(chm)
chm_smoothed <- terra::focal(chm_terra, w=3, fun="mean", na.rm=TRUE)
chm <- raster(chm_smoothed)  # Convert back to raster for consistency

# Verify we have data
max_val <- cellStats(chm, "max")
if (is.na(max_val) || max_val == -Inf || max_val <= 0) {
cat(" Warning: CHM has no positive values after generation.\n")
return(NULL)
}

# Convert final raster to SpatRaster for consistency before returning
chm_terra <- terra::rast(chm)

cat(sprintf(" CHM: %d x %d pixels, Height range: %.2f to %.2f m\n",
ncol(chm_terra), nrow(chm_terra),
terra::global(chm_terra, "min", na.rm = TRUE)[1,1],
terra::global(chm_terra, "max", na.rm = TRUE)[1,1]))

return(chm_terra)
}

cat("\nStep 3: Processing each timepoint...\n")

# --- ADDED: Convert terra template to raster for lidR compatibility ---
# lidR's rasterize_* functions expect a RasterLayer object from the 'raster' package
template_raster_legacy <- raster(template_raster)

# Check if CHM already exists
if (file.exists(output_chm)) {
cat(sprintf(" CHM already exists, skipping: %s\n", output_chm))
chm <- terra::rast(output_chm)
cat(sprintf(" Loaded existing CHM: %d x %d pixels\n", ncol(chm), nrow(chm)))
cat(sprintf(" CHM height range: %.2f to %.2f meters\n",
terra::global(chm, "min", na.rm = TRUE)[1,1],
terra::global(chm, "max", na.rm = TRUE)[1,1]))
} else {
# Process each timepoint to generate CHMs
chm_list <- list()

for (i in 1:length(las_files)) {
  chm_result <- process_timepoint(las_files[i], template_raster_legacy, dtm, mean_z_error)
  if (!is.null(chm_result)) {
    chm_list[[length(chm_list) + 1]] <- chm_result
  }
}

cat(sprintf("\nStep 4: Averaging %d CHMs across timepoints...\n", length(chm_list)))

if (length(chm_list) == 0) {
stop("No valid CHMs were generated from any timepoint!")
}

# Average the CHMs
if (length(chm_list) == 1) {
final_chm <- chm_list[[1]]
cat(" Using single CHM (only one timepoint processed successfully)\n")
} else {
chm_stack <- terra::rast(chm_list)
final_chm <- terra::app(chm_stack, mean, na.rm = TRUE)
}

cat(sprintf(" Final CHM: %d x %d pixels, Height range: %.2f to %.2f m\n",
ncol(final_chm), nrow(final_chm),
terra::global(final_chm, "min", na.rm = TRUE)[1,1],
terra::global(final_chm, "max", na.rm = TRUE)[1,1]))

cat(sprintf(" Saving averaged CHM: %s\n", output_chm))
dir.create(dirname(output_chm), showWarnings = FALSE, recursive = TRUE)
terra::writeRaster(final_chm, output_chm, overwrite = TRUE)
}

cat(sprintf("\nProcessing complete!\n"))
cat(sprintf(" Averaged CHM (%.3f m): %s\n", target_resolution, output_chm))
cat(sprintf(" Based on %d timepoints: %s\n", length(las_files),
paste(basename(tools::file_path_sans_ext(las_files)), collapse = ", ")))