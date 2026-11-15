# Offgrid Demo Photos

Drag these files onto the photo panel, paste a copied image with Cmd+V / Ctrl+V,
or use Choose photo. Pasting the image itself works; pasting a website URL does
not download it.

## Suggested Order

| File | Use |
| --- | --- |
| `01-forest-road.jpg` | Start here. Rehearsed photo for the live road-cover demonstration. |
| `04-forest-road-covered.png` | Same photo with a manual, opaque road cover. Compare with a fresh live analysis of `01`, not its historical saved result. |
| `02-open-road.jpg` | A recorded failure case. The historical estimate misses by roughly 1,617 km. |
| `03-winding-road.jpg` | Another recorded example, with roughly 131 km historical error. |
| `05-canal-street.jpg` | Amsterdam canal scene. New, untested live input, not an evaluation example. |
| `06-pedestrian-street.jpg` | Kyoto pedestrian street. New, untested live input, not an evaluation example. |

For the recorded reveal and known-location button, select the matching thumbnail
inside the app. Importing any file through paste, drop, or Choose photo treats
it as a new input: ground truth is unknown to the app and inference is live.

## More Photos

The pack now contains 18 images. Files 07-18 are additional, untested public
photos for live uploads:

| File | Scene |
| --- | --- |
| `07-lisbon-tram.jpg` | Lisbon tram and narrow street |
| `08-san-francisco-crosswalk.jpg` | San Francisco crosswalk from above; a harder, less typical street-view input |
| `09-toronto-street.jpg` | Toronto intersection and buildings |
| `10-paris-colorful-street.jpg` | Colorful houses on Rue Cremieux, Paris |
| `11-london-houses.jpg` | London brick terraces |
| `12-seoul-hanok-street.jpg` | Traditional buildings in Seoul |
| `13-singapore-chinatown.jpg` | Singapore Chinatown shopping street |
| `14-new-york-brownstones.jpg` | New York brownstone facades |
| `15-sydney-terraces.jpg` | Sydney terrace houses |
| `16-cape-town-colorful-street.jpg` | Colorful Wale Street buildings, Cape Town |
| `17-istanbul-beyoglu.jpg` | Istanbul tram and street scene |
| `18-copenhagen-waterfront.jpg` | Nyhavn waterfront, Copenhagen |

These place labels come from the image sources, not from the model. No model
predictions have been measured on these new photos. Try Lisbon, Seoul, Cape
Town, or Copenhagen first for visually distinctive scenes, without assuming the
model will get them right.

All public-image author credits, original source URLs and individual license
links are in `CREDITS.csv` and `manifest.json`; source metadata receipts are
stored beside each image. New copies are resized, JPEG re-encoded and stripped
of EXIF/GPS metadata. Retain each image's attribution and license when sharing;
share-alike requirements still apply where specified.

No embedded EXIF/GPS metadata is present in these copies. Visual details remain.
The first three are deterministic evaluation display exports, not cherry-picked
successes; their saved predictions were generated on the original source pixels.

## Attribution And Reuse

- Files 01-03: OSV-5M, copied from this project's frozen `experiment-v2` exports.
  Original image IDs, known coordinates and source references are in
  `manifest.json`. File 04 is an edited derivative of file 01. Respect upstream
  OSV-5M / source-image terms; these are not relicensed as project software.
- File 05: **Marco Ober**, *Reguliersgracht, Amsterdam*, via Wikimedia Commons,
  [source](https://commons.wikimedia.org/wiki/File:Reguliersgracht,_Amsterdam.jpg),
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
- File 06: **Basile Morin**, *Pedestrian road with pavements and paper umbrellas,
  Higashiyama-ku, Kyoto, Japan, early morning*, via Wikimedia Commons,
  [source](https://commons.wikimedia.org/wiki/File:Pedestrian_road_with_pavements_and_paper_umbrellas,_Higashiyama-ku,_Kyoto,_Japan,_early_morning.jpg),
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

Files 05-06 were resized, re-encoded as JPEG, and stripped of metadata. They are
distributed under the same CC BY-SA 4.0 license. Retain attribution and license
when redistributing these images or adaptations. Source metadata receipts are
included beside them.

Google Maps links in `manifest.json` refer to the known dataset coordinates,
not model estimates. The app's coordinate links open each actual revealed
estimate or known location in a new Google Maps tab; opening a link shares
those coordinates with Google. No Google Maps API key is needed.
