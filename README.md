# AI Recommendation Logic

## Project Overview

AI Recommendation Logic is a simple, content-based recommendation system written in Python. It asks the user for their interests (for example `sci-fi, drama`), compares those interests with the attributes of every item in a movie catalog, and displays the best matches along with the reason each one was recommended.

The project focuses on the fundamentals of personalization: matching a user profile with item attributes using pure similarity logic, before moving on to more complex techniques such as neural collaborative filtering. It was built as Project 3 of the DecodeLabs Industrial Training programme (Batch 2026, Artificial Intelligence track).

## Objectives

- Take user input in the form of choices or interests.
- Match user preferences against item attributes using logic and similarity.
- Display ranked, relevant recommendations to the user.
- Show a similarity score for every recommended item.
- Allow users to rate items so the system learns their preferences.
- Demonstrate logic building, pattern matching and core recommendation concepts.

## Key Features

- **Interest input with validation:** text is cleaned (case, spaces, duplicates) and unknown words are reported instead of crashing the program.
- **Typo correction:** small spelling mistakes are fixed automatically, so `romence` becomes `romance`.
- **Similarity scoring:** every movie receives a match percentage between 0% and 100%.
- **Ranked recommendations:** the top 5 results are shown with a visual bar and a "because you like..." explanation.
- **Rating system:** 1 to 5 star ratings update the user's profile. High ratings strengthen tags, low ratings weaken them or mark them as tags to avoid.
- **Cold-start fallback:** if nothing matches, the most popular movies are suggested.
- **No repeats:** movies the user has already rated are never recommended again.
- **Built-in demo mode and unit tests:** 15 tests cover similarity, input handling and recommendation behaviour.

## Technologies Used

- **Language:** Python 3 (3.8 or newer)
- **Libraries:** Python standard library only (`difflib`, `sys`, `unittest`). No external packages are needed.
- **Interface:** command-line (terminal)
- **Techniques:** content-based filtering, weighted Jaccard similarity, tag-weight learning from ratings

## How the Recommendation Logic Works

1. **Data representation.** Each movie is described by a set of tags (genres and themes) and a popularity score.
2. **User profile.** The interests the user enters become tags, each starting with a weight of 1.0.
3. **Similarity calculation.** Each movie is scored against the profile using weighted Jaccard similarity:

   ```
   score = weight of shared tags / weight of all tags in either set
   ```

   With equal weights this is the standard Jaccard formula, `|A ∩ B| / |A ∪ B|`.

   *Example:* a user likes `sci-fi, drama`. For *Interstellar* (`sci-fi, drama, adventure, space`), 2 tags are shared out of 4 in total, so the score is 2 / 4 = **50%**.

4. **Ranking.** Movies with a score above zero are sorted from highest to lowest. Ties are broken by popularity, and the top 5 are displayed.
5. **Learning from ratings.**
   - 4 to 5 stars: the movie's tags are strengthened and added to the profile.
   - 3 stars: no change.
   - 1 to 2 stars: tags the user liked are weakened, and unfamiliar tags are recorded as ones to avoid, which lowers the score of similar movies.
6. **Fallback.** If no movie matches, the most popular unseen movies are recommended.

## Installation Requirements

- Python 3.8 or newer. Check your version with:

  ```bash
  python --version
  ```

- No additional packages need to be installed.

## How to Run the Project

1. Download or clone the project and open a terminal in the project folder.
2. Start the interactive recommender:

   ```bash
   python recommender.py
   ```

3. Enter your interests separated by commas, for example `sci-fi, drama`.
4. Use the menu to rate a movie (`r`), enter new interests (`n`) or quit (`q`).

Other commands:

```bash
python recommender.py --demo               # scripted demo, no typing needed
python -m unittest -v test_recommender     # run the unit tests
```

On some systems, use `python3` instead of `python`.

## Project Folder Structure

```
AI-Recommendation-Logic/
├── recommender.py          # Dataset, similarity logic, rating system and command-line app
├── test_recommender.py     # 15 unit tests
└── README.md               # Project documentation
```

## Author

**Nwonu Ikenna Basil**
