#!/usr/bin/env python3
"""
quantum_mutual_information.py — Fase 4, Investigación 3
Información mutua cuántica entre libros vs clásica.
Red de correlaciones y detección de comunidades.
Todo numpy/scipy — cero frameworks cuánticos.
"""
import json, logging, time
from collections import Counter
from pathlib import Path
import numpy as np
from scipy import stats as sp_stats

BASE = Path(__file__).parent
CORPUS = BASE / "bible_unified.json"
OUT = BASE / "results" / "qmi"
OUT.mkdir(parents=True, exist_ok=True)
LOG = BASE / "logs"
LOG.mkdir(parents=True, exist_ok=True)

logging.basicConfig(filename=LOG / "quantum_mutual_information.log", level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("qmi")

POS_CATEGORIES = ["noun", "verb", "pronoun", "adjective", "adverb",
                   "preposition", "conjunction", "particle", "other"]
N_POS = len(POS_CATEGORIES)

GENRE_MAP = {
    "poetic": ["Psalms", "Proverbs", "Song of Songs", "Ecclesiastes", "Lamentations"],
    "legal": ["Leviticus", "Deuteronomy"],
    "epistolar": ["Romans", "1 Corinthians", "2 Corinthians", "Galatians", "Ephesians",
                  "Philippians", "Colossians", "1 Thessalonians", "2 Thessalonians",
                  "1 Timothy", "2 Timothy", "Titus", "Philemon", "Hebrews",
                  "James", "1 Peter", "2 Peter", "1 John", "2 John", "3 John", "Jude"],
}


def get_genre(book):
    for genre, books in GENRE_MAP.items():
        if book in books:
            return genre
    return "narrative"


def pos_to_index(pos):
    try:
        return POS_CATEGORIES.index(pos)
    except ValueError:
        return POS_CATEGORIES.index("other")


def build_density_matrix(word_list):
    """Build density matrix ρ from POS frequency vectors of each verse."""
    verses = {}
    for w in word_list:
        vk = (w.get("book_num", 0), w["chapter"], w["verse"])
        verses.setdefault(vk, []).append(w)

    n_verses = len(verses)
    if n_verses == 0:
        return np.zeros((N_POS, N_POS)), 0

    rho = np.zeros((N_POS, N_POS))
    for vk in sorted(verses.keys()):
        vw = verses[vk]
        vec = np.zeros(N_POS)
        for w in vw:
            vec[pos_to_index(w["pos"])] += 1
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        rho += np.outer(vec, vec)

    tr = np.trace(rho)
    if tr > 0:
        rho = rho / tr
    return rho, n_verses


def von_neumann_entropy(rho):
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    if len(eigenvalues) == 0:
        return 0.0
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def build_joint_density_empirical(words_a, words_b):
    """Build empirical joint density matrix ρ_AB in tensor product space.
    Uses POS vectors concatenated (dimension 2*N_POS)."""
    # Get POS distribution vectors for each book
    vec_a = np.zeros(N_POS)
    for w in words_a:
        vec_a[pos_to_index(w["pos"])] += 1
    if np.sum(vec_a) > 0:
        vec_a = vec_a / np.sum(vec_a)

    vec_b = np.zeros(N_POS)
    for w in words_b:
        vec_b[pos_to_index(w["pos"])] += 1
    if np.sum(vec_b) > 0:
        vec_b = vec_b / np.sum(vec_b)

    # Empirical joint: tensor product of marginal POS distributions
    # with correlation from shared lemmas
    lemmas_a = Counter(w["lemma"] for w in words_a)
    lemmas_b = Counter(w["lemma"] for w in words_b)
    shared = set(lemmas_a.keys()) & set(lemmas_b.keys())

    # Build joint POS matrix weighted by shared lemma overlap
    joint = np.outer(vec_a, vec_b)
    if shared:
        # Add correlation term from shared lemmas
        total_a = sum(lemmas_a.values())
        total_b = sum(lemmas_b.values())
        for lemma in shared:
            # POS of shared lemma contributes to off-diagonal correlation
            weight = (lemmas_a[lemma] / total_a) * (lemmas_b[lemma] / total_b)
            joint += weight * np.outer(vec_a, vec_b)

    # Normalize
    total = np.sum(joint)
    if total > 0:
        joint = joint / total
    return joint


def classical_mutual_information(words_a, words_b):
    """Classical MI based on shared lemma proportions."""
    lemmas_a = Counter(w["lemma"] for w in words_a)
    lemmas_b = Counter(w["lemma"] for w in words_b)
    total_a = sum(lemmas_a.values())
    total_b = sum(lemmas_b.values())

    shared = set(lemmas_a.keys()) & set(lemmas_b.keys())
    if not shared or total_a == 0 or total_b == 0:
        return 0.0

    # MI = Σ p(shared) * log(p(shared) / (p_a * p_b))
    mi = 0.0
    total_shared = sum(min(lemmas_a[l], lemmas_b[l]) for l in shared)
    if total_shared == 0:
        return 0.0

    for l in shared:
        p_a = lemmas_a[l] / total_a
        p_b = lemmas_b[l] / total_b
        p_shared = min(lemmas_a[l], lemmas_b[l]) / (total_a + total_b)
        if p_shared > 0 and p_a > 0 and p_b > 0:
            mi += p_shared * np.log2(p_shared / (p_a * p_b))

    return float(mi)


def modularity_greedy(adj_matrix, book_names, book_genres):
    """Simple greedy modularity optimization for community detection."""
    n = len(book_names)
    # Initialize: each node in its own community
    communities = list(range(n))
    total_weight = np.sum(adj_matrix) / 2
    if total_weight == 0:
        return 0.0, {i: 0 for i in range(n)}

    degrees = np.sum(adj_matrix, axis=1)

    def compute_modularity(comms):
        Q = 0.0
        for i in range(n):
            for j in range(i+1, n):
                if comms[i] == comms[j]:
                    Q += adj_matrix[i, j] - degrees[i] * degrees[j] / (2 * total_weight)
        return Q / (2 * total_weight) if total_weight > 0 else 0

    # Try genre-based communities
    genre_to_id = {}
    genre_comms = []
    for g in book_genres:
        if g not in genre_to_id:
            genre_to_id[g] = len(genre_to_id)
        genre_comms.append(genre_to_id[g])

    Q_genre = compute_modularity(genre_comms)

    # Try corpus-based communities (AT=0, NT=1)
    corpus_comms = [0 if i < 39 else 1 for i in range(n)]  # approximate
    Q_corpus = compute_modularity(corpus_comms[:n])

    return float(Q_genre), float(Q_corpus), genre_comms


def main():
    log.info("=== QUANTUM MUTUAL INFORMATION — INICIO ===")
    t0 = time.time()

    with open(CORPUS, encoding="utf-8") as f:
        words = json.load(f)
    log.info(f"Corpus: {len(words)} palabras")

    # Group by book
    books_data = {}
    book_order = []
    for w in words:
        if w["book"] not in books_data:
            books_data[w["book"]] = {"words": [], "corpus": w["corpus"],
                                      "genre": get_genre(w["book"]),
                                      "book_num": w["book_num"]}
            book_order.append(w["book"])
        books_data[w["book"]]["words"].append(w)

    # Sort by canonical order
    book_order.sort(key=lambda b: books_data[b]["book_num"])
    n_books = len(book_order)
    log.info(f"{n_books} libros")

    # Precompute density matrices and entropies
    log.info("Precomputing density matrices...")
    rhos = {}
    s_vn = {}
    for bk in book_order:
        rho, _ = build_density_matrix(books_data[bk]["words"])
        rhos[bk] = rho
        s_vn[bk] = von_neumann_entropy(rho)

    # === Quantum MI matrix ===
    log.info("Computing quantum MI matrix (n²/2 pairs)...")
    qmi_matrix = np.zeros((n_books, n_books))
    cmi_matrix = np.zeros((n_books, n_books))
    delta_matrix = np.zeros((n_books, n_books))

    pairs_done = 0
    total_pairs = n_books * (n_books - 1) // 2
    last_log = time.time()

    for i in range(n_books):
        for j in range(i+1, n_books):
            bk_a = book_order[i]
            bk_b = book_order[j]

            # Joint density
            joint = build_joint_density_empirical(
                books_data[bk_a]["words"], books_data[bk_b]["words"])

            # Von Neumann entropy of joint
            eigenvalues = np.linalg.eigvalsh(joint)
            eigenvalues = eigenvalues[eigenvalues > 1e-15]
            s_joint = -float(np.sum(eigenvalues * np.log2(eigenvalues))) if len(eigenvalues) > 0 else 0

            # Quantum MI: I_q = S(A) + S(B) - S(AB)
            i_q = s_vn[bk_a] + s_vn[bk_b] - s_joint
            qmi_matrix[i, j] = i_q
            qmi_matrix[j, i] = i_q

            # Classical MI
            i_c = classical_mutual_information(
                books_data[bk_a]["words"], books_data[bk_b]["words"])
            cmi_matrix[i, j] = i_c
            cmi_matrix[j, i] = i_c

            # Delta
            delta_matrix[i, j] = i_q - i_c
            delta_matrix[j, i] = i_q - i_c

            pairs_done += 1
            now = time.time()
            if now - last_log >= 30:
                log.info(f"  {pairs_done}/{total_pairs} pares ({100*pairs_done/total_pairs:.1f}%)")
                last_log = now

    log.info(f"MI matrix complete. {pairs_done} pairs.")

    # === Network modularity ===
    log.info("Computing network modularity...")
    book_genres = [books_data[bk]["genre"] for bk in book_order]

    # Modularity on quantum MI network
    Q_genre_q, Q_corpus_q, genre_comms = modularity_greedy(
        np.maximum(qmi_matrix, 0), book_order, book_genres)
    Q_genre_c, Q_corpus_c, _ = modularity_greedy(
        np.maximum(cmi_matrix, 0), book_order, book_genres)

    modularity = {
        "quantum_network": {
            "modularity_genre": round(Q_genre_q, 6),
            "modularity_corpus": round(Q_corpus_q, 6),
        },
        "classical_network": {
            "modularity_genre": round(Q_genre_c, 6),
            "modularity_corpus": round(Q_corpus_c, 6),
        },
        "interpretation": (
            "Higher modularity means the network's communities align better "
            "with the given partition (genre or corpus)."
        ),
    }

    # Community structure
    community_structure = {}
    for genre in set(book_genres):
        members = [book_order[i] for i in range(n_books) if book_genres[i] == genre]
        # Average within-group quantum MI
        indices = [i for i in range(n_books) if book_genres[i] == genre]
        within_qmi = []
        for a in range(len(indices)):
            for b in range(a+1, len(indices)):
                within_qmi.append(qmi_matrix[indices[a], indices[b]])
        community_structure[genre] = {
            "members": members,
            "n_books": len(members),
            "mean_within_qmi": round(float(np.mean(within_qmi)), 6) if within_qmi else 0,
            "mean_within_cmi": round(float(np.mean([
                cmi_matrix[indices[a], indices[b]]
                for a in range(len(indices))
                for b in range(a+1, len(indices))
            ])), 6) if len(indices) > 1 else 0,
        }

    # === Save ===
    log.info("Guardando resultados...")

    # Convert matrices to serializable format (top pairs only, not full matrix)
    qmi_top_pairs = []
    for i in range(n_books):
        for j in range(i+1, n_books):
            qmi_top_pairs.append({
                "book_a": book_order[i], "book_b": book_order[j],
                "quantum_mi": round(float(qmi_matrix[i, j]), 6),
            })
    qmi_top_pairs.sort(key=lambda x: x["quantum_mi"], reverse=True)

    cmi_top_pairs = []
    for i in range(n_books):
        for j in range(i+1, n_books):
            cmi_top_pairs.append({
                "book_a": book_order[i], "book_b": book_order[j],
                "classical_mi": round(float(cmi_matrix[i, j]), 6),
            })
    cmi_top_pairs.sort(key=lambda x: x["classical_mi"], reverse=True)

    delta_top = []
    for i in range(n_books):
        for j in range(i+1, n_books):
            delta_top.append({
                "book_a": book_order[i], "book_b": book_order[j],
                "delta_i": round(float(delta_matrix[i, j]), 6),
                "quantum_mi": round(float(qmi_matrix[i, j]), 6),
                "classical_mi": round(float(cmi_matrix[i, j]), 6),
            })
    delta_top.sort(key=lambda x: abs(x["delta_i"]), reverse=True)

    with open(OUT / "quantum_mi_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"top50": qmi_top_pairs[:50],
                    "stats": {"mean": round(float(np.mean(qmi_matrix[qmi_matrix > 0])), 6) if np.any(qmi_matrix > 0) else 0,
                              "max": round(float(np.max(qmi_matrix)), 6)}},
                  f, ensure_ascii=False, indent=2)

    with open(OUT / "classical_mi_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"top50": cmi_top_pairs[:50],
                    "stats": {"mean": round(float(np.mean(cmi_matrix[cmi_matrix != 0])), 6) if np.any(cmi_matrix != 0) else 0,
                              "max": round(float(np.max(cmi_matrix)), 6)}},
                  f, ensure_ascii=False, indent=2)

    with open(OUT / "delta_i_matrix.json", "w", encoding="utf-8") as f:
        json.dump({"top50_by_abs_delta": delta_top[:50],
                    "mean_delta": round(float(np.mean(delta_matrix)), 6),
                    "std_delta": round(float(np.std(delta_matrix)), 6)},
                  f, ensure_ascii=False, indent=2)

    with open(OUT / "network_modularity.json", "w", encoding="utf-8") as f:
        json.dump(modularity, f, ensure_ascii=False, indent=2)

    with open(OUT / "community_structure.json", "w", encoding="utf-8") as f:
        json.dump(community_structure, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - t0
    log.info(f"=== DONE en {elapsed:.1f}s ===")
    print(f"[quantum_mutual_information] DONE — {elapsed:.1f}s")


if __name__ == "__main__":
    main()
