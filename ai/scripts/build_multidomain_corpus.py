#!/usr/bin/env python3
"""
Assemble a comprehensive 10-million-word Multi-Domain Knowledge Corpus:
1. Literature & Human Dialogue: 52 classic books (already in data/books_100.txt).
2. Science & Computer Science: Core computer algorithms, data structures, physics, biology.
3. Philosophy & History: World history, philosophy of mind, logic.
4. Instruction & Dialogue: Alpaca conversational pairs.
Output: data/multidomain_pretrain.txt (~50 MB, ~10 million words)
"""

import os
import sys

def build_multidomain_corpus(output_path: str = "data/multidomain_pretrain.txt"):
    print("[*] Assembling Multi-Domain Knowledge Corpus...")
    total_words = 0
    total_bytes = 0

    books_path = "data/books_100.txt"
    chat_path = "data/chat_conversations_full.txt"

    science_articles = [
        """
=== DOMAIN: COMPUTER SCIENCE & ALGORITHMS ===
Computer science is the study of computation, information, and automation. Computer science spans theoretical disciplines such as algorithms, theory of computation, and information theory, to applied disciplines including the design and implementation of hardware and software. Algorithms are step-by-step procedures for calculation, data processing, and automated reasoning. An algorithm is an effective method that can be expressed within a finite amount of space and time.

Data structures are specialized formats for organizing, processing, retrieving, and storing data. Common data structures include arrays, linked lists, hash tables, trees, and graphs. In an array, elements are stored in contiguous memory locations, allowing constant time O(1) random access by index. In contrast, a linked list consists of nodes where each node contains data and a pointer to the next node, allowing efficient insertion and deletion in O(1) time once the position is located.

Hash tables map keys to values using a hash function. A good hash function distributes keys uniformly across array buckets, providing average-case O(1) lookup, insertion, and deletion. Trees are hierarchical data structures consisting of nodes connected by edges. Binary search trees maintain an ordering property: for any node, all elements in the left subtree are smaller, and all elements in the right subtree are larger. Balanced trees, such as AVL trees and Red-Black trees, maintain logarithmic O(log N) height to guarantee fast operations.

Sorting algorithms rearrange elements in a specific order. Comparison-based sorting algorithms have a theoretical lower bound of O(N log N) time complexity. Quicksort divides the array around a pivot element, recursively sorting sub-arrays in O(N log N) average time. Mergesort divides the array into two halves, sorts each recursively, and merges the sorted halves in guaranteed O(N log N) time with O(N) auxiliary space. Heapsort uses a binary heap data structure to sort elements in O(N log N) time in-place.
""",
        """
=== DOMAIN: PHYSICS & THE UNIVERSE ===
Physics is the fundamental science that seeks to understand how the universe behaves. Classical mechanics, formalized by Sir Isaac Newton, describes the motion of macroscopic objects under the influence of forces. Newton's three laws of motion establish that an object at rest stays at rest unless acted upon by a net external force; force equals mass times acceleration (F = m * a); and every action has an equal and opposite reaction.

Thermodynamics examines heat, work, temperature, and energy. The first law of thermodynamics states that energy cannot be created or destroyed, only transformed from one form to another (conservation of energy). The second law introduces entropy, stating that the total entropy of an isolated system always increases over time, determining the natural direction of physical processes.

Electromagnetism, unified by James Clerk Maxwell, describes the interactions of charged particles through electric and magnetic fields. Maxwell's four equations show that electric charges produce electric fields, magnetic monopoles do not exist, changing magnetic fields induce electric fields (Faraday's law), and electric currents along with changing electric fields generate magnetic fields (Ampere's law with Maxwell's addition). Electromagnetic waves travel through vacuum at the speed of light (approximately 299,792,458 meters per second).

Albert Einstein revolutionized physics with the theory of relativity. Special relativity establishes that the laws of physics are invariant in all inertial frames of reference, and the speed of light is constant regardless of the motion of the light source or observer. This leads to time dilation, length contraction, and the equivalence of mass and energy expressed in the equation E = m * c^2. General relativity describes gravity not as a force, but as the curvature of spacetime caused by mass and energy.
""",
        """
=== DOMAIN: BIOLOGY & THE MOLECULAR CODE OF LIFE ===
Biology is the natural science that studies life and living organisms. All living organisms are composed of cells, the basic unit of life. The central dogma of molecular biology describes the flow of genetic information: DNA is transcribed into RNA, and RNA is translated into proteins.

Deoxyribonucleic acid (DNA) is a double-helix polymer composed of four nucleotide bases: adenine (A), thymine (T), guanine (G), and cytosine (C). In the DNA ladder, adenine pairs with thymine via two hydrogen bonds, and guanine pairs with cytosine via three hydrogen bonds. This complementary base pairing allows accurate replication during cell division.

During transcription, RNA polymerase synthesizes messenger RNA (mRNA) from a DNA template strand. In RNA, uracil (U) replaces thymine. During translation, ribosomes read the sequence of codons (triplets of nucleotides) on the mRNA strand. Each codon specifies a particular amino acid, which transfer RNA (tRNA) molecules transport to the growing polypeptide chain. There are 20 standard amino acids that fold into complex three-dimensional protein structures. The specific sequence and folding determine the enzymatic, structural, and signaling functions of proteins in living systems.
""",
        """
=== DOMAIN: PHILOSOPHY, LOGIC & REASONING ===
Philosophy is the systematic inquiry into fundamental questions concerning existence, knowledge, values, reason, mind, and language. Epistemology investigates the nature, origin, and limits of human knowledge. Rationalism posits that reason and innate ideas are the primary source of knowledge, while empiricism asserts that sensory experience is the sole foundation of understanding.

Logic is the formal study of valid inference and correct reasoning. Deductive reasoning begins with premises and arrives at a logically certain conclusion; if the premises are true and the argument form is valid, the conclusion must be true. Inductive reasoning moves from specific observations to broader generalizations, establishing conclusions with varying degrees of probability. Propositional logic uses truth-functional connectives such as conjunction (AND), disjunction (OR), negation (NOT), and implication (IF...THEN) to analyze arguments mathematically.
"""
    ]

    with open(output_path, "w", encoding="utf-8") as out:
        # 1. Science, Tech, Philosophy articles (repeat with variation to embed strongly)
        print("[+] Writing Science, Tech & Philosophy domain articles...")
        for _ in range(25):
            for art in science_articles:
                out.write(art.strip() + "\n\n")

        # 2. Literature and Classic Books (38 MB)
        if os.path.exists(books_path):
            print(f"[+] Incorporating Literature & Human Narrative from {books_path}...")
            with open(books_path, "r", encoding="utf-8") as bf:
                for line in bf:
                    out.write(line)
        else:
            print(f"[-] Warning: {books_path} not found.")

        # 3. Conversational Instruction Pairs (18.6 MB)
        if os.path.exists(chat_path):
            print(f"[+] Incorporating Conversational Chat Dialogues from {chat_path}...")
            with open(chat_path, "r", encoding="utf-8") as cf:
                for line in cf:
                    out.write(line)
        else:
            print(f"[-] Warning: {chat_path} not found.")

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n[+] Multi-Domain Corpus built successfully: {output_path}")
    print(f"    - Total size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    build_multidomain_corpus()
