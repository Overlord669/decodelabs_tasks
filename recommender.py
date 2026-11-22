"""
AI Recommendation Logic  -  DecodeLabs Project 3
================================================

A content-based recommendation engine built with pure similarity logic
(no external libraries needed).

How it works
------------
1. Every item (movie) is described by a set of tags (genres / themes).
2. The user tells us their interests (tags).
3. We compare the user's profile with every item using
   *weighted Jaccard similarity*:

       score = weight of shared tags / weight of all tags in either set

4. Items are ranked by score and the best ones are shown, together with
   the reason (which tags matched).
5. The user can rate movies 1-5. Ratings adjust the weight of the tags in
   the profile (likes strengthen tags, dislikes weaken or penalise them),
   so the next recommendations get smarter.

Run it:
    python recommender.py          # interactive mode
    python recommender.py --demo   # scripted demo, no typing needed
"""

import difflib
import sys

# --------------------------------------------------------------------------
# 1. DATASET
# --------------------------------------------------------------------------
# popularity (1-10) is only used for tie-breaking and for the cold-start
# fallback (when a user gives us nothing we can match).
MOVIES = {
    "Inception":              {"tags": {"sci-fi", "thriller", "action", "mind-bending"}, "popularity": 9},
    "Interstellar":           {"tags": {"sci-fi", "drama", "adventure", "space"}, "popularity": 9},
    "The Matrix":             {"tags": {"sci-fi", "action", "mind-bending"}, "popularity": 9},
    "Blade Runner 2049":      {"tags": {"sci-fi", "drama", "mystery"}, "popularity": 7},
    "Arrival":                {"tags": {"sci-fi", "drama", "mystery", "space"}, "popularity": 7},
    "The Notebook":           {"tags": {"romance", "drama"}, "popularity": 8},
    "La La Land":             {"tags": {"romance", "musical", "drama"}, "popularity": 8},
    "Pride and Prejudice":    {"tags": {"romance", "drama", "classic"}, "popularity": 7},
    "The Dark Knight":        {"tags": {"action", "crime", "thriller", "superhero"}, "popularity": 10},
    "Avengers: Endgame":      {"tags": {"action", "superhero", "adventure", "sci-fi"}, "popularity": 10},
    "Mad Max: Fury Road":     {"tags": {"action", "adventure", "sci-fi"}, "popularity": 8},
    "John Wick":              {"tags": {"action", "crime", "thriller"}, "popularity": 8},
    "Get Out":                {"tags": {"horror", "thriller", "mystery"}, "popularity": 8},
    "A Quiet Place":          {"tags": {"horror", "thriller", "sci-fi"}, "popularity": 7},
    "The Conjuring":          {"tags": {"horror", "mystery", "supernatural"}, "popularity": 7},
    "Superbad":               {"tags": {"comedy", "teen"}, "popularity": 7},
    "The Grand Budapest Hotel": {"tags": {"comedy", "adventure", "crime", "quirky"}, "popularity": 7},
    "Home Alone":             {"tags": {"comedy", "family", "classic"}, "popularity": 8},
    "Toy Story":              {"tags": {"animation", "family", "comedy", "adventure"}, "popularity": 9},
    "Spirited Away":          {"tags": {"animation", "fantasy", "adventure", "family"}, "popularity": 8},
    "The Lord of the Rings":  {"tags": {"fantasy", "adventure", "action", "epic"}, "popularity": 10},
    "Harry Potter":           {"tags": {"fantasy", "adventure", "family", "supernatural"}, "popularity": 9},
    "The Shawshank Redemption": {"tags": {"drama", "crime", "classic"}, "popularity": 10},
    "Parasite":               {"tags": {"thriller", "drama", "comedy", "crime"}, "popularity": 9},
    "Gone Girl":              {"tags": {"thriller", "mystery", "crime", "drama"}, "popularity": 8},
    "Planet Earth":           {"tags": {"documentary", "nature", "adventure"}, "popularity": 8},
}


def all_tags(catalog=MOVIES):
    """Every tag that exists in the catalog."""
    tags = set()
    for info in catalog.values():
        tags |= info["tags"]
    return tags


# --------------------------------------------------------------------------
# 2. INPUT HANDLING
# --------------------------------------------------------------------------
def parse_interests(raw, known_tags):
    """
    Turn text like ' Sci-Fi, DRAMA ,romence ' into
    (valid_tags, unknown_words). Small typos are corrected with difflib.
    """
    valid, unknown = [], []
    for word in raw.replace(";", ",").split(","):
        word = word.strip().lower()
        if not word:
            continue
        if word in known_tags:
            match = word
        else:
            close = difflib.get_close_matches(word, known_tags, n=1, cutoff=0.75)
            match = close[0] if close else None
        if match is None:
            unknown.append(word)
        elif match not in valid:
            valid.append(match)
    return valid, unknown


# --------------------------------------------------------------------------
# 3. SIMILARITY LOGIC
# --------------------------------------------------------------------------
def weighted_jaccard(user_weights, item_tags):
    """
    Jaccard similarity where each tag can have a different weight.
    With every weight = 1.0 this is plain Jaccard: |A n B| / |A u B|.
    Returns a number between 0 (nothing in common) and 1 (identical).
    """
    shared = set(user_weights) & item_tags
    union = set(user_weights) | item_tags
    if not union:
        return 0.0
    top = sum(user_weights[t] for t in shared)
    bottom = sum(user_weights.get(t, 1.0) for t in union)
    return top / bottom if bottom else 0.0


class UserProfile:
    """Stores what we know about the user: interests, ratings, dislikes."""

    def __init__(self, interests=()):
        self.weights = {tag: 1.0 for tag in interests}   # liked tags
        self.disliked = {}                                # tag -> strength
        self.rated = {}                                   # movie -> rating

    def rate(self, title, rating, catalog=MOVIES):
        """
        Learn from a 1-5 rating.
        4-5 stars  -> strengthen the movie's tags (and add them to interests)
        3 stars    -> neutral, nothing changes
        1-2 stars  -> weaken tags we liked, and remember tags to avoid
        """
        if title not in catalog:
            raise KeyError(f"Unknown movie: {title}")
        if not 1 <= rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        self.rated[title] = rating
        delta = (rating - 3) * 0.25
        for tag in catalog[title]["tags"]:
            if rating >= 4:
                self.weights[tag] = min(3.0, self.weights.get(tag, 0.75) + delta)
                self.disliked.pop(tag, None)
            elif rating <= 2:
                if tag in self.weights:
                    self.weights[tag] = max(0.1, self.weights[tag] + delta)
                else:
                    self.disliked[tag] = min(1.0, self.disliked.get(tag, 0.0) - delta)

    def score(self, tags):
        """Final match score for an item with the given tags (0 to 1)."""
        base = weighted_jaccard(self.weights, tags)
        penalty = 0.0
        if tags:
            penalty = 0.3 * sum(self.disliked.get(t, 0.0) for t in tags) / len(tags)
        return max(0.0, base - penalty)


# --------------------------------------------------------------------------
# 4. RECOMMENDATION
# --------------------------------------------------------------------------
def recommend(profile, top_n=5, catalog=MOVIES):
    """
    Return a list of (title, score, matched_tags) sorted best-first.
    Movies the user already rated are skipped.
    Falls back to the most popular movies if nothing matches (cold start).
    """
    results = []
    for title, info in catalog.items():
        if title in profile.rated:
            continue
        score = profile.score(info["tags"])
        if score > 0:
            matched = sorted(set(profile.weights) & info["tags"])
            results.append((title, score, matched))

    results.sort(key=lambda r: (-r[1], -catalog[r[0]]["popularity"], r[0]))

    if not results:
        fallback = sorted(
            (t for t in catalog if t not in profile.rated),
            key=lambda t: (-catalog[t]["popularity"], t),
        )
        return [(t, 0.0, []) for t in fallback[:top_n]]
    return results[:top_n]


# --------------------------------------------------------------------------
# 5. DISPLAY
# --------------------------------------------------------------------------
def show(recs):
    if not recs:
        print("  Nothing left to recommend - you've rated everything!")
        return
    for i, (title, score, matched) in enumerate(recs, 1):
        bar = "#" * round(score * 20)
        if matched:
            print(f"  {i}. {title:<26} {score:>4.0%} match  [{bar:<20}]")
            print(f"     because you like: {', '.join(matched)}")
        else:
            print(f"  {i}. {title:<26} (popular pick)")


# --------------------------------------------------------------------------
# 6. COMMAND-LINE APP
# --------------------------------------------------------------------------
def ask(prompt):
    try:
        return input(prompt).strip()
    except EOFError:
        return "q"


def choose_interests(tags):
    print("\nAvailable interests:")
    print("  " + ", ".join(sorted(tags)))
    while True:
        raw = ask("\nEnter your interests, separated by commas (or 'q' to quit): ")
        if raw.lower() == "q":
            return None
        valid, unknown = parse_interests(raw, tags)
        if unknown:
            print(f"  (Ignored unknown: {', '.join(unknown)})")
        if valid:
            print(f"  Using: {', '.join(valid)}")
            return valid
        print("  Please enter at least one interest from the list above.")


def rate_flow(profile):
    title = ask("Movie title to rate: ")
    lookup = {t.lower(): t for t in MOVIES}
    key = title.lower()
    if key not in lookup:
        close = difflib.get_close_matches(key, lookup, n=1, cutoff=0.6)
        if not close:
            print("  Movie not found in the catalog.")
            return
        key = close[0]
    movie = lookup[key]
    raw = ask(f"Rating for '{movie}' (1-5): ")
    try:
        profile.rate(movie, int(raw))
    except ValueError:
        print("  Please enter a whole number from 1 to 5.")
        return
    print(f"  Saved. Your profile has been updated from '{movie}'.")


def interactive():
    print("=" * 58)
    print("  AI RECOMMENDATION ENGINE  |  DecodeLabs Project 3")
    print("=" * 58)
    tags = all_tags()
    interests = choose_interests(tags)
    if interests is None:
        return
    profile = UserProfile(interests)

    while True:
        print("\nTop recommendations for you:")
        show(recommend(profile))
        print("\n  [r] rate a movie   [n] new interests   [q] quit")
        choice = ask("> ").lower()
        if choice == "r":
            rate_flow(profile)
        elif choice == "n":
            new = choose_interests(tags)
            if new:
                profile = UserProfile(new)
        elif choice == "q":
            print("Thanks for using the recommender. Goodbye!")
            return


def demo():
    """Scripted run that shows every feature without typing."""
    print("=== DEMO ===\n")
    tags = all_tags()
    valid, unknown = parse_interests("Sci-Fi, drama ,  romence, pizza", tags)
    print(f"Input: 'Sci-Fi, drama ,  romence, pizza'")
    print(f"  -> understood {valid}, ignored {unknown}\n")

    profile = UserProfile(["sci-fi", "drama"])
    print("1) Recommendations for interests: sci-fi, drama")
    show(recommend(profile))

    print("\n2) User rates 'Interstellar' 5 stars and 'The Notebook' 1 star")
    profile.rate("Interstellar", 5)
    profile.rate("The Notebook", 1)
    show(recommend(profile))

    print("\n3) Cold start: user gives no usable interests")
    show(recommend(UserProfile([]), top_n=3))


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        interactive()
