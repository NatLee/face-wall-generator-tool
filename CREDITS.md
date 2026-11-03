# Credits

Everything below that is not this project's own code comes from somewhere else. Here is where,
who made it and under which licence.

## Sample faces and portraits

The images in [`faces/`](faces/) and [`examples/portraits/`](examples/portraits/) all come from
Wikimedia Commons and are in the public domain:

- The photos are official NASA astronaut portraits. Works of the U.S. federal government are not
  covered by copyright in the United States (Commons tag `PD-USGov-NASA`).
- The paintings and the print are old enough that their copyright has long expired, and Commons
  treats faithful photos of them as public domain too (`PD-Art`).

Public domain covers the copyright only. The people in the photos still have personality rights,
and NASA asks that its images are not used to suggest that NASA or the astronauts endorse
anything. Using them as test data for a face model is fine; putting them in an advert is not.

How they were made: we downloaded the 960 px wide thumbnails that Commons serves for each file.
For `faces/` we cropped each one around the face with this tool's own `--crop-faces` code and
resized it to 256 x 256. For `examples/portraits/` we only shrank the long side to 480 px.

| File | Who / what | Author | Source on Wikimedia Commons | Licence |
|---|---|---|---|---|
| `victor-glover.jpg` (both folders) | Victor J. Glover, NASA astronaut | Robert Markowitz, NASA | [File:Victor J. Glover official portrait.jpg](https://commons.wikimedia.org/wiki/File:Victor_J._Glover_official_portrait.jpg) | Public domain (PD-USGov-NASA) |
| `jonny-kim.jpg` | Jonny Kim, NASA astronaut | Bill Stafford, NASA | [File:Jonny Kim official portrait.jpg](https://commons.wikimedia.org/wiki/File:Jonny_Kim_official_portrait.jpg) | Public domain (PD-USGov-NASA) |
| `anil-menon.jpg` | Anil Menon, NASA astronaut | Robert Markowitz, NASA | [File:Anil Menon portrait.jpg](https://commons.wikimedia.org/wiki/File:Anil_Menon_portrait.jpg) | Public domain (PD-USGov-NASA) |
| `mike-hopkins.jpg` (both folders) | Mike Hopkins, NASA astronaut | Norah Moran, NASA | [File:Mike Hopkins official portrait 2020.jpg](https://commons.wikimedia.org/wiki/File:Mike_Hopkins_official_portrait_2020.jpg) | Public domain (PD-USGov-NASA) |
| `kimiya-yui.jpg` | Kimiya Yui, astronaut (official NASA photo) | NASA | [File:Kimiya Yui NASA official portrait.jpg](https://commons.wikimedia.org/wiki/File:Kimiya_Yui_NASA_official_portrait.jpg) | Public domain (PD-USGov-NASA) |
| `jessica-watkins.jpg` | Jessica Watkins, NASA astronaut | Bill Stafford, NASA | [File:Jessica Watkins official portrait.jpg](https://commons.wikimedia.org/wiki/File:Jessica_Watkins_official_portrait.jpg) | Public domain (PD-USGov-NASA) |
| `jasmin-moghbeli.jpg` | Jasmin Moghbeli, NASA astronaut | Bill Stafford, NASA | [File:Jasmin Moghbeli official portrait.jpg](https://commons.wikimedia.org/wiki/File:Jasmin_Moghbeli_official_portrait.jpg) | Public domain (PD-USGov-NASA) |
| `zena-cardman.jpg` | Zena Cardman, NASA astronaut | Bill Stafford, NASA | [File:Zena Cardman official portrait.jpg](https://commons.wikimedia.org/wiki/File:Zena_Cardman_official_portrait.jpg) | Public domain (PD-USGov-NASA) |
| `shannon-walker.jpg` | Shannon Walker, NASA astronaut | Norah Moran, NASA | [File:Shannon Walker official portrait 2020.jpg](https://commons.wikimedia.org/wiki/File:Shannon_Walker_official_portrait_2020.jpg) | Public domain (PD-USGov-NASA) |
| `mona-lisa.jpg` | *Mona Lisa* (1503-1506), painting | Leonardo da Vinci | [File:Mona Lisa, by Leonardo da Vinci, from C2RMF retouched.jpg](https://commons.wikimedia.org/wiki/File:Mona_Lisa,_by_Leonardo_da_Vinci,_from_C2RMF_retouched.jpg) | Public domain (PD-Art) |
| `girl-with-a-pearl-earring.jpg` (both folders) | *Girl with a Pearl Earring* (1665), painting | Johannes Vermeer | [File:1665 Girl with a Pearl Earring.jpg](https://commons.wikimedia.org/wiki/File:1665_Girl_with_a_Pearl_Earring.jpg) | Public domain (PD-Art) |
| `sharaku-otani-oniji.jpg` (both folders) | Actor Ōtani Oniji III as Edobei (1794), woodblock print | Toshusai Sharaku | [File:Toshusai Sharaku- Otani Oniji, 1794.jpg](https://commons.wikimedia.org/wiki/File:Toshusai_Sharaku-_Otani_Oniji,_1794.jpg) | Public domain (PD-Art) |

Files marked "both folders" are in `faces/` as a cropped tile and in `examples/portraits/` as the
uncropped picture; the others are only in `faces/`.

## Face detector

[`face_wall/models/face_detection_yunet_2026may.onnx`](face_wall/models/) is YuNet from the
[OpenCV model zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
by Shiqi Yu and contributors, under the MIT licence
([`face_wall/models/LICENSE-YuNet`](face_wall/models/LICENSE-YuNet)). Paper: Wu, Peng and Yu,
"YuNet: A tiny millisecond-level face detector", *Machine Intelligence Research* 20(5), 2023.
