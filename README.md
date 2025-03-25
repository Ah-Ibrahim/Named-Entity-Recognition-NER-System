# Named Entity Recognition (NER) System

## 🚀 Project Overview  
The **Named Entity Recognition (NER) System** is a machine learning-based solution that identifies and classifies named entities in text. It supports multiple entity types and implements two state-of-the-art probabilistic models:  

- **Hidden Markov Model (HMM)**  
- **Maximum Entropy Markov Model (MEMM)**  

This system can be used in various Natural Language Processing (NLP) applications, such as search engines, chatbots, and information extraction tools.

## 📌 Features  
✅ **Multi-Class Entity Recognition**: Detects **Persons (PER), Organizations (ORG), Locations (LOC), Miscellaneous (MISC), and Non-Entities (O)**.  
✅ **Dual Model Implementation**: Compare HMM and MEMM for NER tasks.  
✅ **Pre-Trained & Custom Training**: Works with provided datasets or allows custom training.  
✅ **Modular & Scalable**: Structured to allow further improvements and feature additions.  

## 📊 Named Entity Recognition Example  
### **Input Sentence:**  
> *"Steve Jobs founded Apple with Steve Wozniak."*  

### **NER Output:**  
```plaintext
Steve     -> PER  
Jobs      -> PER  
founded   -> O  
Apple     -> ORG  
with      -> O  
Steve     -> PER  
Wozniak   -> PER  
```
