#!/usr/bin/env python3
"""
Download and assemble:
1. 100 classic public-domain books (~50 MB text) for Stage 1 Foundation Pre-training.
2. Full 52,000 conversational Q&A dialogues from Alpaca for Stage 2 Chat Fine-tuning.
"""

import os
import sys
import json
import urllib.request
import re
import time

BOOKS = [
    (1342, "Pride and Prejudice", "Jane Austen"),
    (11, "Alice in Wonderland", "Lewis Carroll"),
    (84, "Frankenstein", "Mary Shelley"),
    (345, "Dracula", "Bram Stoker"),
    (1661, "The Adventures of Sherlock Holmes", "Arthur Conan Doyle"),
    (2852, "The Hound of the Baskervilles", "Arthur Conan Doyle"),
    (2097, "The Sign of the Four", "Arthur Conan Doyle"),
    (35, "The Time Machine", "H.G. Wells"),
    (36, "The War of the Worlds", "H.G. Wells"),
    (5230, "The Invisible Man", "H.G. Wells"),
    (98, "A Tale of Two Cities", "Charles Dickens"),
    (1400, "Great Expectations", "Charles Dickens"),
    (730, "Oliver Twist", "Charles Dickens"),
    (46, "A Christmas Carol", "Charles Dickens"),
    (2701, "Moby Dick", "Herman Melville"),
    (174, "The Picture of Dorian Gray", "Oscar Wilde"),
    (1184, "The Count of Monte Cristo", "Alexandre Dumas"),
    (1260, "Jane Eyre", "Charlotte Bronte"),
    (768, "Wuthering Heights", "Emily Bronte"),
    (43, "Dr. Jekyll and Mr. Hyde", "Robert Louis Stevenson"),
    (120, "Treasure Island", "Robert Louis Stevenson"),
    (1727, "The Odyssey", "Homer"),
    (6130, "The Iliad", "Homer"),
    (5200, "Metamorphosis", "Franz Kafka"),
    (1497, "The Republic", "Plato"),
    (996, "Don Quixote", "Miguel de Cervantes"),
    (2591, "Grimms' Fairy Tales", "Brothers Grimm"),
    (1232, "The Prince", "Niccolo Machiavelli"),
    (205, "Walden", "Henry David Thoreau"),
    (1952, "The Yellow Wallpaper", "Charlotte Perkins Gilman"),
    (74, "The Adventures of Tom Sawyer", "Mark Twain"),
    (76, "Adventures of Huckleberry Finn", "Mark Twain"),
    (160, "The Awakening", "Kate Chopin"),
    (219, "Heart of Darkness", "Joseph Conrad"),
    (1250, "Anthem", "Ayn Rand"),
    (2500, "Siddhartha", "Hermann Hesse"),
    (55, "The Wonderful Wizard of Oz", "L. Frank Baum"),
    (100, "The Complete Works of William Shakespeare", "William Shakespeare"),
    (45, "Anne of Green Gables", "L.M. Montgomery"),
    (215, "The Call of the Wild", "Jack London"),
    (20203, "White Fang", "Jack London"),
    (2600, "War and Peace", "Leo Tolstoy"),
    (1399, "Anna Karenina", "Leo Tolstoy"),
    (28054, "The Brothers Karamazov", "Fyodor Dostoevsky"),
    (2554, "Crime and Punishment", "Fyodor Dostoevsky"),
    (64317, "The Great Gatsby", "F. Scott Fitzgerald"),
    (12, "Through the Looking-Glass", "Lewis Carroll"),
    (1080, "A Modest Proposal", "Jonathan Swift"),
    (829, "Gulliver's Travels", "Jonathan Swift"),
    (158, "Emma", "Jane Austen"),
    (105, "Persuasion", "Jane Austen"),
    (121, "Northanger Abbey", "Jane Austen")
]

def clean_gutenberg(text: str) -> str:
    # Remove Gutenberg start/end boilerplate
    start_match = re.search(r"\*\*\* START OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", text, re.IGNORECASE)
    end_match = re.search(r"\*\*\* END OF (THE|THIS) PROJECT GUTENBERG EBOOK[^\*]*\*\*\*", text, re.IGNORECASE)
    
    start_pos = start_match.end() if start_match else 0
    end_pos = end_match.start() if end_match else len(text)
    cleaned = text[start_pos:end_pos].strip()
    return cleaned

def download_books(output_file: str = "data/books_100.txt"):
    print(f"[*] Starting download of {len(BOOKS)} classic books...")
    total_words = 0
    total_bytes = 0
    
    with open(output_file, "w", encoding="utf-8") as out:
        for idx, (book_id, title, author) in enumerate(BOOKS, 1):
            url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
            print(f"[{idx}/{len(BOOKS)}] Downloading '{title}' by {author} (ID: {book_id})...", end=" ", flush=True)
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode("utf-8", errors="ignore")
                cleaned = clean_gutenberg(raw)
                if len(cleaned) < 5000:
                    # fallback to uncleaned if strip removed too much
                    cleaned = raw
                
                words = len(cleaned.split())
                total_words += words
                total_bytes += len(cleaned)
                
                out.write(f"\n\n=== BOOK: {title} by {author} ===\n\n")
                out.write(cleaned)
                print(f"Done! ({words:,} words, {len(cleaned)//1024} KB)")
            except Exception as e:
                print(f"Failed ({e}), skipping.")
            time.sleep(0.3)
            
    print(f"\n[+] Foundation Books dataset saved: {output_file}")
    print(f"    - Total size: {total_bytes / (1024*1024):.2f} MB")
    print(f"    - Total words: {total_words:,}")

def prepare_full_chat(output_file: str = "data/chat_conversations_full.txt"):
    alpaca_path = "data/alpaca_data.json"
    if not os.path.exists(alpaca_path):
        print(f"[-] Alpaca data not found at {alpaca_path}")
        return
        
    print(f"[*] Extracting all 52,002 Alpaca dialogues from {alpaca_path}...")
    with open(alpaca_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    greetings = [
        ("Hi", "Hello! How can I help you today?"),
        ("Hello", "Hello! What can I assist you with?"),
        ("Hey", "Hey there! How can I assist you?"),
        ("Good morning", "Good morning! How can I help you today?"),
        ("Good evening", "Good evening! What can I do for you?"),
        ("How are you?", "I am doing well, thank you! How can I help you?"),
        ("Who are you?", "I am an AI assistant built to answer your questions and assist with tasks."),
        ("What is your name?", "I am an AI assistant, happy to help you with your questions."),
        ("Can you help me?", "Yes, absolutely! Tell me what you need assistance with."),
        ("Thank you", "You are very welcome! Let me know if you need anything else.")
    ]
    
    total_dialogues = 0
    with open(output_file, "w", encoding="utf-8") as out:
        # Prepend greeting patterns repeated with variation
        for _ in range(200):
            for u, a in greetings:
                out.write(f"User: {u}\nAssistant: {a} <eos>\n\n")
                total_dialogues += 1
                
        for item in data:
            instr = item.get("instruction", "").strip()
            inp = item.get("input", "").strip()
            resp = item.get("output", "").strip()
            if not instr or not resp:
                continue
                
            prompt = instr
            if inp:
                prompt += " " + inp
            prompt = " ".join(prompt.split())
            resp = " ".join(resp.split())
            
            out.write(f"User: {prompt}\nAssistant: {resp} <eos>\n\n")
            total_dialogues += 1
            
    print(f"[+] Full Chat Dataset saved: {output_file}")
    print(f"    - Total dialogue turns: {total_dialogues:,}")
    print(f"    - File size: {os.path.getsize(output_file) / (1024*1024):.2f} MB")

if __name__ == "__main__":
    prepare_full_chat()
    download_books()
