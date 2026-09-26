# Learning & Coding Guidelines

## 1. Teaching Methodology: Example First, Terms Later
- **Never introduce technical terms/jargon upfront.**
- Always start with an intuitive, visual, concrete example or numbers first.
- Explain what is physically or logically happening step-by-step.
- Only introduce the formal names / industry terms **at the very end**, explaining: *"By the way, people in the AI industry call this concept X."*

## 2. Language Choice: C++ Primary, Python Side-by-Side
- Kushagra understands **C++** much better.
- **Always write and explain code in clean, modern C++ first.**
- Provide the Python equivalent or industry library comparison side-by-side so Kushagra also learns how it maps to industry practice.

## 3. First-Principles Implementation
- Build concepts from scratch (arrays, loops, simple math) before introducing third-party abstractions.

## 4. Build Output Directory
- **Always compile C++ binaries into a `build/` directory** (e.g. `g++ ... -o build/<binary_name> && ./build/<binary_name>`), never into the source folder.
- Ensure `build/` is gitignored so the repository stays clean.
