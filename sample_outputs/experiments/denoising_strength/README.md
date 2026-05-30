# Denoising Strength Experiment

## Objective
Evaluate the effect of denoising strength on image transformation quality in Stable Diffusion img2img workflows.

## Input
- Saree product image
- Model: epicrealismXL_pureFix
- Sampler: Euler a
- Steps: 32
- CFG Scale: 6

## Experiments

### Denoising 0.15
Preserved nearly all original details and structure.

### Denoising 0.20
Minor improvements while maintaining design consistency.

### Denoising 0.50
Introduced noticeable creative modifications and texture changes.

### Denoising 0.80
Significant transformation with major deviations from the source image.

## Conclusion

Lower denoising values preserve product fidelity while higher values encourage creative reinterpretation. For fashion-product enhancement workflows, denoising values between 0.15 and 0.30 provided the best balance between realism and design preservation.
