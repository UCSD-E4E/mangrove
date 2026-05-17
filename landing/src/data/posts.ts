export type Block =
  | { type: 'p'; text: string }
  | { type: 'h2'; text: string }
  | { type: 'h3'; text: string }
  | { type: 'pullquote'; text: string }
  | { type: 'image'; src?: string; caption: string; alt: string }

export interface Post {
  slug: string
  date: string
  tag: string
  title: string
  excerpt: string
  readTime: string
  body: Block[]
}

const TAG_COLORS: Record<string, string> = {
  Research: '#3d6b4a',
  ML: '#4a6b8a',
  Engineering: '#6b4a3d',
  Science: '#6b5a3d',
}
export { TAG_COLORS }

export const POSTS: Post[] = [
  {
    slug: 'super-resolution-pipeline',
    date: 'Apr 2026',
    tag: 'ML',
    title: 'Recovering Boundaries: Our Super-Resolution Pipeline',
    excerpt:
      'Standard 10m Sentinel-2 predictions blur the tidal channels and fragmented edges that define mangrove ecosystem health. Our hybrid transformer-CNN model upsamples predictions 16 times, resolving structure that was previously invisible.',
    readTime: '8 min read',
    body: [
      {
        type: 'p',
        text: 'A Sentinel-2 pixel is 10 meters across. That is large enough to detect a mangrove forest. It is not large enough to resolve a tidal channel, measure the width of a mangrove fringe, or detect the fragmented edges where forest is breaking apart under pressure. For regional-scale monitoring, 10 meters is workable. For understanding ecosystem structure and health at the level of individual stands and canopy gaps, it is not.',
      },
      {
        type: 'p',
        text: 'This is the problem super-resolution is designed to address.',
      },
      {
        type: 'h2',
        text: 'What Super-Resolution Actually Does',
      },
      {
        type: 'p',
        text: 'Super-resolution is the task of recovering fine spatial detail from low-resolution input. In the context of our pipeline, this means taking the 10-meter segmentation output from our SegFormer model and upsampling it to 0.6 meters per pixel, a factor of 16 in each spatial dimension.',
      },
      {
        type: 'p',
        text: 'It is important to be precise about what this means. The model cannot invent spatial information that was never captured by the satellite sensor. What it can do is learn, from paired high-resolution and low-resolution training examples, how fine-scale structure tends to relate to coarse-scale patterns. Tidal channels follow the topographic low points in a mangrove flat. Canopy gaps cluster in predictable ways relative to the forest edge. These regularities are learnable, and a well-trained model can use them to produce upsampled predictions that are substantially more accurate at boundaries than naive bilinear interpolation.',
      },
      {
        type: 'h2',
        text: 'The Architecture',
      },
      {
        type: 'p',
        text: 'We use a hybrid architecture that combines transformer and convolutional components. The transformer component operates at the coarser spatial scale and captures long-range dependencies: the relationship between a mangrove fringe on one side of a tidal channel and the corresponding edge on the other side, for example. The convolutional component handles the local texture and edge reconstruction that the transformer, operating at patch granularity, is less well suited to.',
      },
      {
        type: 'p',
        text: 'The design reflects a practical observation from the super-resolution literature: transformers are good at reasoning about global structure, while convolutions are good at recovering local high-frequency detail. For mangrove upsampling, you need both. The channel networks and canopy gaps that define ecosystem structure require global context to be correctly reconstructed, while the sharp edges between classes require local convolution to render cleanly.',
      },
      {
        type: 'image',
        src: '/blog/sentinel2_10m.png',
        alt: 'Sentinel-2 10m resolution imagery of Florida mangroves',
        caption: 'Input: Sentinel-2 at native 10m resolution. Individual pixels are large enough to see the block structure clearly at this scale.',
      },
      {
        type: 'image',
        src: '/blog/naip_0.6m.png',
        alt: 'High-resolution 0.6m imagery of the same Florida mangrove area',
        caption: 'Target: 0.6m per pixel. Tidal channels, canopy gaps, and mangrove fringes become individually resolvable at this resolution.',
      },
      {
        type: 'h2',
        text: 'Training Data',
      },
      {
        type: 'p',
        text: 'Training a super-resolution model requires paired examples: the same geographic area captured at both the input resolution and the target resolution. For our Florida training data, we use NAIP aerial imagery at approximately one meter per pixel as the high-resolution reference. NAIP is collected by the United States Department of Agriculture and covers the continental United States at consistent quality and resolution.',
      },
      {
        type: 'p',
        text: 'The training procedure is straightforward in principle: downsample the NAIP imagery to simulate 10-meter input, train the model to recover the original NAIP resolution from the downsampled version, and evaluate against the true NAIP imagery. The challenge is in the alignment: Sentinel-2 and NAIP are captured at different times, by different sensors, with different atmospheric conditions. Getting the spatial registration accurate enough to train a useful model required careful preprocessing.',
      },
      {
        type: 'h2',
        text: 'What Changes at 0.6 Meters',
      },
      {
        type: 'p',
        text: 'At 10-meter resolution, a tidal channel narrower than one pixel width is invisible. At 0.6 meters, a channel two meters wide is clearly resolved. This matters for carbon accounting because the channel network structure determines how water and sediment move through the forest, which in turn affects productivity and carbon accumulation rates.',
      },
      {
        type: 'p',
        text: 'At 10-meter resolution, the boundary between mangrove and adjacent land cover is a zone of ambiguous mixed pixels several pixels wide. At 0.6 meters, the boundary is a sharp line. This matters for monitoring because early-stage mangrove loss typically affects the forest edge first. Accurate edge delineation is a prerequisite for detecting incremental loss before it accumulates to a scale visible at coarser resolution.',
      },
      {
        type: 'pullquote',
        text: 'The goal was never to make satellite imagery look like drone imagery. The goal was to make it good enough to answer the questions that conservation depends on.',
      },
      {
        type: 'h2',
        text: 'Limitations and Next Steps',
      },
      {
        type: 'p',
        text: 'Super-resolution is not a free lunch. The upsampled predictions are more detailed than the raw satellite output, but they carry uncertainty that the coarser output does not. In regions where the satellite input is spectrally ambiguous, the super-resolution model will produce confident-looking but potentially incorrect fine-scale predictions. Quantifying that uncertainty and communicating it to end users is an active area of development.',
      },
      {
        type: 'p',
        text: 'NAIP coverage is limited to the United States, which constrains where we can train and validate the super-resolution model directly. Extending the approach to other regions requires either collecting equivalent high-resolution reference imagery or developing methods that can transfer super-resolution capability learned in one region to another where only coarser reference data is available.',
      },
    ],
  },
  {
    slug: 'continual-learning',
    date: 'Mar 2026',
    tag: 'Engineering',
    title: 'Continual Learning Across Regions',
    excerpt:
      'A model trained on Florida tends to forget Florida when fine-tuned on Yucatan. We fixed this with a replay buffer that keeps 10 percent of prior region samples in every training batch.',
    readTime: '6 min read',
    body: [
      {
        type: 'p',
        text: 'Mangroves in Florida look different from mangroves in the Sundarbans. They grow in different soils, alongside different species, under different atmospheric conditions, and against completely different coastal backgrounds. A model trained on Florida will generalize poorly to Indonesia, and a model trained on Indonesia will struggle with Madagascar. This is expected, and it is solvable by fine-tuning on each new region in sequence.',
      },
      {
        type: 'p',
        text: 'What is less obvious, and considerably more problematic, is what happens to Florida performance after you fine-tune on Brazil.',
      },
      {
        type: 'h2',
        text: 'Catastrophic Forgetting',
      },
      {
        type: 'p',
        text: 'Neural networks learn by adjusting their weights in response to new data. When you fine-tune a network on data from a new region, those adjustments are driven entirely by the new region\'s statistics. The weights that encoded knowledge about the previous region get overwritten. The model forgets.',
      },
      {
        type: 'p',
        text: 'This phenomenon is called catastrophic forgetting, and it is one of the central unsolved problems in machine learning. A neural network, unlike a human expert, does not accumulate knowledge across tasks. It replaces old knowledge with new knowledge every time it is trained on something different.',
      },
      {
        type: 'p',
        text: 'In our case, we measured a drop of roughly 15 percentage points in Florida IoU after fine-tuning the model on Yucatan data, even though the Yucatan training set was much smaller. The model had effectively unlearned what it knew about Florida in order to fit the new distribution.',
      },
      {
        type: 'pullquote',
        text: 'A model that can map the Yucatan but no longer maps Florida correctly is not a global monitoring tool. It is just a different regional model.',
      },
      {
        type: 'h2',
        text: 'The Replay Buffer Approach',
      },
      {
        type: 'p',
        text: 'The solution we implemented is a replay buffer, sometimes called experience replay. The core idea is simple: when fine-tuning on a new region, do not train exclusively on new data. Instead, mix a small fraction of examples from all previously seen regions into every training batch.',
      },
      {
        type: 'p',
        text: 'In practice, we maintain a buffer of training samples from every region the model has been trained on. Each training batch is assembled by our ReplayMixDataset class: 90 percent of each batch comes from the current region being learned, and 10 percent is sampled uniformly from the replay buffer containing prior region data. This fraction is a hyperparameter, but we found that 10 percent was sufficient to preserve prior performance without significantly slowing convergence on the new region.',
      },
      {
        type: 'h2',
        text: 'Why It Works',
      },
      {
        type: 'p',
        text: 'Catastrophic forgetting happens because gradient updates during fine-tuning pull the model\'s weights toward the new distribution and away from the old one. By including old examples in every batch, you ensure that gradients from the old distribution are present in every weight update. The model cannot fully forget a region because it is continuously reminded of it.',
      },
      {
        type: 'p',
        text: 'This is a simplified version of approaches used in continual learning research, which has produced more sophisticated methods involving separate memory modules, knowledge distillation, and architectural changes to protect prior knowledge. For our use case, the straightforward replay buffer is effective and operationally simple. We do not need to store the full training datasets from prior regions: a randomly sampled subset representing roughly 10 percent of the prior data is sufficient.',
      },
      {
        type: 'image',
        src: undefined,
        alt: 'Chart comparing Florida IoU before fine-tuning, after fine-tuning without replay, and after fine-tuning with replay buffer',
        caption: 'Florida validation IoU across three conditions. Fine-tuning on Yucatan without replay causes a sharp drop in Florida performance. The replay buffer recovers most of that loss while still achieving strong Yucatan results.',
      },
      {
        type: 'h2',
        text: 'The Six Regions We Train On',
      },
      {
        type: 'p',
        text: 'Our current pipeline covers six ecologically distinct regions, trained in sequence: Florida, which includes the South Florida coast and the Everglades; Brazil, covering the Amazon estuary and northern coast; Indonesia, spanning a wide archipelago with highly fragmented mangrove distribution; Madagascar and Mozambique, characterized by estuarine channels and tidal mudflats along the East African coast; North Australia, covering the Kimberley coast and Gulf of Carpentaria; and the East India and Bangladesh Sundarbans, one of the largest continuous mangrove systems on Earth and one of the hardest to classify due to dense forest canopy and high adjacent cropland.',
      },
      {
        type: 'p',
        text: 'Each region introduces new spectral conditions, new surrounding land cover classes, and new tidal regimes. Without the replay buffer, performance on Florida degrades measurably after training on Brazil, and degrades further with each subsequent region. With the replay buffer active, the model retains most of its original performance on each prior region while still adapting to the new one.',
      },
      {
        type: 'p',
        text: 'This makes the pipeline genuinely scalable. Adding a new region does not require retraining on all prior regions simultaneously. It requires a fine-tuning run on the new region\'s data with a 10 percent replay sample from the buffer. As coverage expands, the cost of each addition stays roughly constant rather than growing with the total number of regions already trained.',
      },
    ],
  },
  {
    slug: 'satellite-embeddings',
    date: 'Jan 2026',
    tag: 'ML',
    title: 'Why Satellite Embeddings Are a Game Changer',
    excerpt:
      'Google\'s DINO-based satellite embeddings compress rich spectral and textural context into 64 dimensions per pixel. Prepending them to our SegFormer input lifted validation IoU by 11 points without adding labeled data.',
    readTime: '8 min read',
    body: [
      {
        type: 'p',
        text: 'The previous post described our 69-band input tiles and mentioned that 64 of those bands are something other than optical satellite channels. This post explains what they are, where they come from, and why adding them changed our model\'s performance more than any other single decision in the pipeline.',
      },
      {
        type: 'h2',
        text: 'The Problem With Raw Spectral Bands',
      },
      {
        type: 'p',
        text: 'Sentinel-2 gives us four usable optical bands at 10-meter resolution: red, green, blue, and near-infrared. These four numbers per pixel describe how much light is reflected in each spectral range. They are the raw material of remote sensing, and they work reasonably well for distinguishing broad land cover categories like forest, water, and urban.',
      },
      {
        type: 'p',
        text: 'The problem is that mangroves are spectrally similar to other types of dense tropical vegetation. A mangrove pixel and a palm plantation pixel can produce nearly identical red, green, blue, and near-infrared reflectance values. Standard vegetation indices like NDVI, which have been used in remote sensing for decades, cannot reliably separate the two. The model needs more information than four spectral bands can provide.',
      },
      {
        type: 'h2',
        text: 'What an Embedding Actually Is',
      },
      {
        type: 'p',
        text: 'An embedding is a compact, learned representation of something complex. In natural language processing, word embeddings encode semantic meaning: words with similar meanings end up close together in the embedding space. In computer vision, image embeddings encode visual content: patches that look similar, or belong to similar semantic categories, end up nearby in the embedding space.',
      },
      {
        type: 'p',
        text: 'Google trained a foundation model on a massive archive of global satellite imagery using DINO, a self-supervised learning technique that does not require labeled data. The model learned to produce a 64-dimensional embedding for every 10-meter pixel by predicting its own representations under different augmentations of the same scene. Because the model was trained on imagery from every type of land cover on Earth, its embeddings capture rich contextual and textural information that goes far beyond what four spectral bands can encode.',
      },
      {
        type: 'pullquote',
        text: 'The embedding does not just describe what color a pixel is. It encodes what kind of place that pixel appears to be, based on everything the model has seen across the planet.',
      },
      {
        type: 'h2',
        text: 'Integrating 68 Channels into SegFormer',
      },
      {
        type: 'p',
        text: 'SegFormer expects a three-channel RGB image. Our model input has 68 channels: four Sentinel-2 optical bands plus 64 embedding dimensions. The label band from the GeoTIFF is excluded from the input entirely; it is the target the model is trained to predict, not a feature it gets to see.',
      },
      {
        type: 'p',
        text: 'Our solution is a lightweight input projection block: a 1x1 convolution that maps the 68-channel input to a hidden representation, followed by a second 1x1 convolution that projects down to the three channels the SegFormer encoder expects. A 1x1 convolution is effectively a learned linear combination of the input channels computed independently at each spatial position. It adds almost no computational overhead but gives the model full flexibility to determine which combinations of optical bands and embedding dimensions are most useful for the segmentation task.',
      },
      {
        type: 'image',
        src: undefined,
        alt: 'Architecture diagram showing the input projection layer feeding into the SegFormer encoder',
        caption: 'The input projection block. A 68-channel input (4 optical + 64 embeddings) is projected to 32 channels and then to 3 channels before entering the standard SegFormer encoder. Both projections are 1x1 convolutions with batch normalization and ReLU.',
      },
      {
        type: 'h2',
        text: 'What the Numbers Say',
      },
      {
        type: 'p',
        text: 'Adding the satellite embeddings lifted our validation intersection-over-union by 11 percentage points compared to training on the four optical bands alone. That is a substantial gain for a single architectural change that added almost no additional labeled training data and minimal extra computation.',
      },
      {
        type: 'p',
        text: 'The improvement was not uniform across classes. The largest gains were on the mangrove class itself and on the distinction between mangrove and other dense vegetation. These are precisely the cases where four optical bands are ambiguous and where the richer contextual information in the embeddings makes the difference.',
      },
      {
        type: 'h2',
        text: 'Why This Matters Beyond Our Project',
      },
      {
        type: 'p',
        text: 'Foundation models trained on global satellite archives represent a shift in how remote sensing research works. Previously, building a good classifier for any specific land cover type required collecting large quantities of labeled training data for that specific task. With strong pretrained embeddings, you can achieve competitive performance with a much smaller labeled dataset because the embeddings already encode most of the visual information the model needs.',
      },
      {
        type: 'p',
        text: 'For conservation applications, where labeled data is expensive to collect and often scarce, this changes what is practically achievable. A team of researchers can now train a useful global mangrove classifier with a fraction of the labeled data that would have been required just a few years ago.',
      },
    ],
  },
  {
    slug: 'scaling-to-satellites',
    date: 'Dec 2025',
    tag: 'Research',
    title: 'Scaling to Satellites: What We Gain and What We Lose',
    excerpt:
      'Moving from drone to Sentinel-2 imagery multiplies coverage but trades spatial detail for global reach. We walk through the trade-offs and why 10m resolution is still useful for coastline-scale monitoring.',
    readTime: '6 min read',
    body: [
      {
        type: 'p',
        text: 'The drone pipeline taught us how to segment mangroves. The satellite pipeline is where we learned how to do it at scale. The two problems look similar on the surface. They are not.',
      },
      {
        type: 'h2',
        text: 'The Resolution Trade-off',
      },
      {
        type: 'p',
        text: 'A drone image at typical survey altitude gives you a pixel every two to five centimeters. An individual prop root is visible. A Sentinel-2 pixel is 10 meters across. At that scale, an entire mangrove stand can occupy just a handful of pixels. Tidal channels that were crisp lines in the drone imagery become single-pixel ambiguities in the satellite image, or disappear entirely.',
      },
      {
        type: 'p',
        text: 'This is not a flaw in Sentinel-2. It is a consequence of the fundamental physics of imaging from 800 kilometers above the surface. What you sacrifice in spatial detail you gain in something no drone can provide: continuous, global, repeat coverage. Sentinel-2 revisits every point on Earth\'s surface every five days. It covers every mangrove coastline on every continent simultaneously. A drone cannot do that in a thousand years of continuous flight.',
      },
      {
        type: 'pullquote',
        text: 'The question was never whether satellite imagery is as good as drone imagery. The question is whether it is good enough to detect the changes that matter, at the scale where those changes are happening.',
      },
      {
        type: 'h2',
        text: 'Why Cloud Cover Is a Serious Problem',
      },
      {
        type: 'p',
        text: 'Mangroves grow in tropical and subtropical regions. These are also among the cloudiest places on Earth. On any given day, a significant fraction of the world\'s mangrove coastlines are obscured by cloud. A single Sentinel-2 acquisition over a tropical site might be entirely unusable.',
      },
      {
        type: 'p',
        text: 'The standard solution is temporal compositing. Instead of relying on any single image, you collect all acquisitions over a multi-month window, filter out cloudy pixels, and combine the remaining observations into a single cloud-free composite. We use Google Earth Engine to do this at scale: pulling all Sentinel-2 Surface Reflectance scenes for a given region over a defined time period, applying a cloud mask using the scene classification layer that comes with each image, and computing a median composite from the remaining observations.',
      },
      {
        type: 'p',
        text: 'The median is important. It is more robust than the mean to residual cloud contamination and atmospheric haze that slips through the cloud mask. A single bright, cloud-affected pixel in a stack of ten observations will not significantly shift the median, but it can substantially distort the mean.',
      },
      {
        type: 'h2',
        text: 'The Tidal Zone Problem',
      },
      {
        type: 'p',
        text: 'Mangroves grow precisely where land meets tidal water. This makes them uniquely difficult to map from satellites. The boundary between mangrove and open water is not fixed. It moves with every tide cycle, shifting by tens to hundreds of meters depending on the local tidal range and coastal slope.',
      },
      {
        type: 'p',
        text: 'When you composite multiple satellite acquisitions taken at different tidal states, the boundary between mangrove and water becomes blurred. A pixel that is underwater during a high-tide acquisition and vegetated during a low-tide acquisition will produce an ambiguous spectral signature in the composite. Standard vegetation indices that work reliably for inland forests become unreliable at the mangrove fringe.',
      },
      {
        type: 'p',
        text: 'We partially mitigate this by filtering acquisitions to a consistent tidal window where possible, and by relying on the learned representations in our model rather than hand-crafted spectral indices. But tidal ambiguity remains one of the hardest open problems in automated mangrove mapping.',
      },
      {
        type: 'h2',
        text: 'What the Training Tiles Contain',
      },
      {
        type: 'p',
        text: 'Each exported tile is a 2048 by 2048 pixel GeoTIFF at 10-meter resolution. The file contains 69 bands in total, but it is important to understand what those bands actually are. The first four are the standard Sentinel-2 optical channels: red, green, blue, and near-infrared. The next 64 are Google\'s DINO-based satellite embeddings, which encode rich spatial and contextual information learned from a global archive of imagery. The final band is the ESA WorldCover land cover label, which is used for training supervision and not passed to the model as an input.',
      },
      {
        type: 'p',
        text: 'In other words, the model sees 68 channels of actual input: four optical bands and 64 learned embeddings. The label band is the target the model is trained to predict. What those embeddings are and why they matter so much is the subject of the next post.',
      },
      {
        type: 'image',
        src: undefined,
        alt: 'Diagram showing the 69-band GeoTIFF structure: 4 optical bands plus 64 embedding channels plus 1 label band',
        caption: 'Structure of a single exported tile. Bands 1 through 4 are Sentinel-2 optical (RGBN). Bands 5 through 68 are 64-dimensional DINO satellite embeddings. Band 69 is the ESA WorldCover label used for supervision only.',
      },
      {
        type: 'p',
        text: 'Our Florida training dataset consists of approximately 100 such tiles covering the South Florida coastline, the Everglades, and surrounding areas. Each 2048-pixel tile is chipped into 512 by 512 patches during training, yielding around 4,000 training chips from the Florida dataset alone.',
      },
      {
        type: 'h2',
        text: 'What We Gain and What We Accept',
      },
      {
        type: 'p',
        text: 'Moving to satellites means accepting lower spatial resolution, tidal ambiguity, and the operational complexity of cloud compositing. What we gain is the ability to monitor coastlines that no field team will ever reach, at a frequency that captures seasonal and inter-annual change as it happens.',
      },
      {
        type: 'p',
        text: 'For conservation purposes, this trade-off is worthwhile. The goal is not to produce a map as detailed as a drone survey. The goal is to produce a map accurate enough to detect where mangroves are being lost, updated frequently enough to act on that information before the loss is irreversible.',
      },
    ],
  },
  {
    slug: 'segmenting-drone-imagery',
    date: 'Nov 2025',
    tag: 'ML',
    title: 'Segmenting Mangroves from Drone Imagery',
    excerpt:
      'We trained SegFormer on two datasets and built a gated ensemble to produce six-class land cover maps at half-meter resolution. Here is how the pieces fit together.',
    readTime: '8 min read',
    body: [
      {
        type: 'p',
        text: 'Every large machine learning project begins with a smaller one. Before we could train a model on satellite imagery, we needed labeled data at a scale where the visual features were clear enough to annotate reliably. That meant drone imagery, and it meant understanding what the drone data could and could not tell us on its own.',
      },
      {
        type: 'h2',
        text: 'The Baja California Dataset',
      },
      {
        type: 'p',
        text: 'Our primary drone training data was collected in Baja California, Mexico. The survey produced 573 labeled samples at 0.5 meters per pixel, each a 512 by 512 tile of three-channel RGB imagery paired with a binary ground-truth mask. The binary annotation is intentionally simple: every pixel is either mangrove or not mangrove. No sub-categories, no ambiguous boundary classes.',
      },
      {
        type: 'p',
        text: 'That simplicity is a strength in one direction and a limitation in another. The binary labels are clean, high-confidence, and straightforward to annotate. They are also insufficient for the kind of land cover analysis that conservation applications actually need. Knowing that a pixel is not mangrove does not tell you whether it is open water, built-up land, bare soil, or agricultural field. Those distinctions matter enormously for understanding the pressures that mangrove ecosystems face.',
      },
      {
        type: 'image',
        src: undefined,
        alt: 'Side-by-side of raw 0.5m drone image from Baja California and its binary mangrove segmentation mask',
        caption: 'A 512x512 tile from the Baja California survey at 0.5m per pixel. The binary mask marks mangrove pixels in green. Non-mangrove pixels are unlabeled at this stage.',
      },
      {
        type: 'h2',
        text: 'The LandCoverAI Dataset',
      },
      {
        type: 'p',
        text: 'To get richer class information, we brought in a second dataset: LandCoverAI, a publicly available benchmark collected over Poland. LandCoverAI contains high-resolution aerial imagery annotated with five land cover classes: background, building, woodland, water, and road. The classes are defined by their visual appearance from above, not by ecological category.',
      },
      {
        type: 'p',
        text: 'Poland and Baja California share almost nothing ecologically. The vegetation types, soil colors, and surrounding land use are entirely different. But they share something more fundamental: the visual geometry of human infrastructure looks similar from above regardless of where it is. A building is a rectangular shape with a sharp roof edge in Gdansk and in Ensenada. A road is a uniform-width linear feature in both places. LandCoverAI gave us a large, well-labeled dataset to train a model specifically for detecting those structures, and that model turned out to be the foundation for a more sophisticated approach.',
      },
      {
        type: 'h2',
        text: 'Choosing SegFormer',
      },
      {
        type: 'p',
        text: 'We evaluated four architectures during development: ResNet-UNet, DenseNet-UNet, DeepLab, and SegFormer. Each represents a different philosophy about how to combine spatial information across an image.',
      },
      {
        type: 'p',
        text: 'ResNet-UNet and DenseNet-UNet use skip connections to preserve fine-grained detail from the encoder through to the decoder. They are reliable, well-understood, and fast to train. DeepLab uses atrous convolutions to capture multi-scale context without collapsing spatial resolution. Both approaches perform well on standard segmentation benchmarks.',
      },
      {
        type: 'p',
        text: 'SegFormer takes a different approach. Its encoder is a hierarchical transformer that computes attention across spatial positions, letting the model relate distant parts of the image to one another. For aerial imagery this matters because the structural regularities of human infrastructure and vegetation canopy span large spatial extents that local convolutions handle poorly. In our experiments, SegFormer produced sharper class boundaries and fewer false positives than the convolutional baselines on both datasets.',
      },
      {
        type: 'h2',
        text: 'The Core Problem: Mangrove Looks Like Woodland',
      },
      {
        type: 'p',
        text: 'Here is where things get complicated. We trained a SegFormer on LandCoverAI to predict the five-class land cover output. When we ran that model on the Baja California drone imagery, it performed well on buildings, roads, and water. But it consistently misclassified mangrove canopy as woodland.',
      },
      {
        type: 'p',
        text: 'This is not a failure of the model. It is the correct behavior given the training data. From above, a dense mangrove canopy and a dense woodland canopy look nearly identical in RGB imagery. Both are green, textured, and continuous. The spectral features that distinguish them in hyperspectral or near-infrared imagery are simply not available in three-channel RGB. A model trained on LandCoverAI has never seen a mangrove, and it has no reason to distinguish mangrove from any other type of dense vegetation.',
      },
      {
        type: 'pullquote',
        text: 'The model was not wrong. It was doing exactly what we asked of it. The problem was that we were asking the wrong question.',
      },
      {
        type: 'h2',
        text: 'The Gated Ensemble',
      },
      {
        type: 'p',
        text: 'The solution was to combine two sources of information that each answer a different question. The LandCoverAI model answers the question of what kind of human or natural land cover occupies each pixel, distinguishing buildings, roads, water, and vegetation. The binary Baja California labels answer the simpler but ecologically essential question of whether each pixel is mangrove.',
      },
      {
        type: 'p',
        text: 'We combine these through a hard gate. The LandCoverAI SegFormer runs inference on the Baja California imagery and produces a five-class prediction for every pixel. We then apply the binary mangrove ground-truth labels as an override: wherever the binary annotation says a pixel is mangrove, we replace the five-class prediction with a sixth class, mangrove, regardless of what the LandCoverAI model predicted.',
      },
      {
        type: 'p',
        text: 'The result is a six-class composite map: background, building, woodland, water, road, and mangrove. The LandCoverAI model handles the human infrastructure classes, which it was trained for. The binary mangrove labels handle the ecological classification, which requires the ground-truth knowledge that the LandCoverAI model cannot provide from RGB alone. The gate is not a learnable component. It is a principled override that routes each pixel to the source of information best positioned to classify it correctly.',
      },
      {
        type: 'image',
        src: undefined,
        alt: 'Diagram of the gated ensemble: LandCoverAI SegFormer predictions merged with binary mangrove GT mask to produce a 6-class output',
        caption: 'The gated ensemble. LandCoverAI SegFormer predicts five classes across the scene. Binary mangrove ground truth overrides any woodland or background prediction within confirmed mangrove extent, producing a six-class composite.',
      },
      {
        type: 'h2',
        text: 'Exporting to CVAT for Label Refinement',
      },
      {
        type: 'p',
        text: 'The six-class composite is not the final output. It is the starting point for human annotation refinement. We export the composite predictions to CVAT, an open source annotation platform, where annotators can review the model-generated masks, correct errors at the boundaries, and resolve ambiguous cases that the ensemble cannot handle with confidence.',
      },
      {
        type: 'p',
        text: 'This workflow reflects a general principle in applied machine learning: the best labels are produced by combining model-generated candidates with human review, not by asking either source to work alone. The model is faster than a human at generating an initial segmentation across hundreds of tiles. The human annotator is better than the model at resolving the edge cases where the classification is genuinely uncertain. The gated ensemble exists specifically to make the human reviewer\'s job tractable by handling the straightforward cases automatically.',
      },
      {
        type: 'h2',
        text: 'Training Setup and Loss Function',
      },
      {
        type: 'p',
        text: 'Images and their corresponding segmentation masks are stored as NumPy arrays and loaded by a custom SegmentationDataset class. Before reaching the model, each image undergoes ImageNet normalization using the same statistics the pretrained encoder was originally trained on. This preserves the low-level feature representations the encoder has already learned and makes fine-tuning substantially more efficient.',
      },
      {
        type: 'p',
        text: 'Class imbalance is a consistent problem in ecological segmentation. In a typical survey tile, mangrove pixels and open water pixels are abundant, while built-up land might account for less than five percent of the image. Standard cross-entropy trained on imbalanced data converges to a shortcut: predict the dominant class everywhere and achieve high pixel accuracy while being entirely useless for the minority classes that actually matter.',
      },
      {
        type: 'p',
        text: 'We address this with a combined loss that sums cross-entropy and Jaccard loss, with a tunable alpha controlling the balance between the two. Jaccard loss directly optimizes intersection-over-union per class, which is naturally more sensitive to minority class performance than cross-entropy alone. For particularly difficult boundary regions, we also experimented with weighted focal loss, which down-weights correctly classified easy pixels and concentrates learning on the hard examples the model keeps getting wrong.',
      },
    ],
  },
  {
    slug: 'why-mangroves-matter',
    date: 'Sep 2025',
    tag: 'Science',
    title: 'Why Mangroves Matter',
    excerpt:
      'Mangroves store 3 to 5 times more carbon per hectare than any terrestrial forest, filter coastal runoff, and buffer communities against storm surge. Yet we are losing them faster than we can map them.',
    readTime: '5 min read',
    body: [
      {
        type: 'p',
        text: 'There is a type of forest that grows with its roots in salt water. It lines the tropical and subtropical coastlines of nearly every continent, forming dense, tangled systems of arching roots and dark canopies that look, from above, almost like green coral. These are mangroves, and they are among the most ecologically important ecosystems on Earth.',
      },
      {
        type: 'p',
        text: 'Most people have never heard of them.',
      },
      {
        type: 'image',
        src: '/blog/mangrove-aerial-canopy.jpg',
        alt: 'Aerial view of mangrove forest canopy with tidal channels',
        caption: 'Mangrove canopy from above. The dark channels running through the green are tidal waterways that flood and drain twice daily.',
      },
      {
        type: 'h2',
        text: 'A Forest That Stores Carbon Backwards',
      },
      {
        type: 'p',
        text: 'When we talk about carbon storage in forests, the instinct is to look up, at trunks and canopies. Mangroves store carbon up there too, but the more important story is underground.',
      },
      {
        type: 'p',
        text: 'Mangrove soils are anoxic. They contain almost no oxygen, which means organic matter decays extraordinarily slowly. Over centuries, layers of roots, leaves, and sediment pile up into a thick, carbon-rich substrate that researchers call blue carbon. A single hectare of mangrove forest can store between 3 and 5 times as much carbon as a hectare of tropical rainforest. And that figure accounts for both the trees and the soil beneath them.',
      },
      {
        type: 'pullquote',
        text: 'A cleared mangrove forest does not just emit carbon. It reverses a geological savings account that took generations to build.',
      },
      {
        type: 'p',
        text: 'This distinction is consequential. When a mangrove forest is cleared, the carbon that escapes is not just what was locked in the living biomass. It is centuries of accumulated soil carbon, released into the atmosphere all at once.',
      },
      {
        type: 'h2',
        text: 'The Storm Wall No One Can Afford to Build',
      },
      {
        type: 'p',
        text: 'Coastal engineers have spent decades constructing hard infrastructure to protect shorelines from storm surge and erosion: seawalls, levees, jetties. Mangroves do the same job for free, and in many cases they do it better.',
      },
      {
        type: 'p',
        text: 'The root networks of mangrove forests dissipate wave energy before it reaches the shore. Research following the 2004 Indian Ocean tsunami showed that settlements shielded by intact mangrove belts suffered substantially less damage than those on exposed coastlines. Studies after cyclones in Bangladesh and storm surges along the Gulf of Mexico have reached similar conclusions. A functioning mangrove belt of 500 meters can reduce incoming wave height by 50 to 70 percent.',
      },
      {
        type: 'p',
        text: 'For the approximately 340 million people who live along tropical coastlines, that is not an abstract statistic. It is the difference between a flooded home and a standing one.',
      },
      {
        type: 'image',
        src: '/blog/mangrove-coastline-belt.jpg',
        alt: 'Aerial view of a mangrove belt separating the ocean from a coastal settlement',
        caption: 'A mangrove belt acting as a natural buffer zone between open water and the coastline.',
      },
      {
        type: 'h2',
        text: 'A Nursery for the Ocean',
      },
      {
        type: 'p',
        text: 'The underwater root structure of a mangrove forest is one of the most productive marine habitats on the planet. Juvenile fish, shrimp, and crustaceans shelter among the roots during the most vulnerable stages of their lives, avoiding predators while they grow large enough to move into open water. Research estimates that more than 80 percent of commercially important tropical fish species depend on mangroves for at least part of their life cycle.',
      },
      {
        type: 'p',
        text: 'Fishermen whose livelihoods depend on coral reef and offshore catches often do not realize how closely those fisheries are connected to the mangrove forests up the coast. When mangroves disappear, fish nurseries disappear with them. The decline in fish populations that follows takes years to materialize, which makes the connection easy to miss until the damage is already done.',
      },
      {
        type: 'h2',
        text: 'The Disappearing Coastline',
      },
      {
        type: 'p',
        text: 'In 1980, the world had an estimated 188 million hectares of mangrove forest. Today, that number is closer to 147 million. Roughly a third of the world\'s mangroves have vanished in less than 50 years.',
      },
      {
        type: 'p',
        text: 'The causes are not mysterious. Shrimp aquaculture has converted entire coastal systems in Southeast Asia, cleared, stocked, and then abandoned when productivity collapses. Urban expansion, agriculture, logging, and pollution account for the rest. What is alarming is the rate: researchers estimate that approximately one percent of remaining mangrove area disappears every year. At that rate, compounded, half of what exists today will be gone within a human lifetime.',
      },
      {
        type: 'h2',
        text: 'You Cannot Protect What You Cannot See',
      },
      {
        type: 'p',
        text: 'Conservation depends on information. To slow or stop this loss, governments and organizations need to know where mangroves are, how much remains, and which areas are under the most pressure. For decades, that knowledge came from ground surveys: field teams walking coastlines, measuring canopy extent, recording coordinates by hand. The work is rigorous but impossibly slow at scale. A team of researchers can survey a few kilometers per week. The world\'s mangrove coastlines stretch for hundreds of thousands of kilometers.',
      },
      {
        type: 'p',
        text: 'Satellite remote sensing changed the equation in principle. But in practice, mapping mangroves from space is harder than it looks. Mangroves occupy the tidal zone, the boundary between land and water, which is precisely where satellite imagery is most difficult to interpret. Water levels shift with the tides, cloud cover over tropical coastlines can persist for weeks, and the spectral signature of mangroves overlaps with other dense vegetation in ways that confuse standard classifiers.',
      },
      {
        type: 'pullquote',
        text: 'This is the problem our team set out to solve: how do you monitor an ecosystem at global scale, continuously, with the resolution needed to actually make decisions?',
      },
    ],
  },
]
