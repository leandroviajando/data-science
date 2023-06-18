# [Introduction to Genomic Technologies](https://www.coursera.org/learn/introduction-genomics)

Humans' DNA is 99.9% identical.

## 1. Molecular Biology

### Genomics

Genomics is the branch of molecular biology concerned with the structure, function, evolution and mapping of genomes.

- **Structure:** what is the biochemical sequence of the genome and what are its physical properties?
- **Function:** what does all the DNA do?
- **Evolution:** how do sequences change over evolutionary time?
- **Mapping:** where are the genes and other interesting bits?

|  | Biology & Genetics | Genomics |
| --- | --- | --- |
Scope: | Targeted studies of one or a few genes | Studies considering all genes in a genome |
| Technology: | Targeted, low-throughput experiments | Global, high-throughput experiments |
| Hard part: | Clever experimental design, painstaking experimentation | Tons of data, uncertainty, computation |

### The Central Dogma of Molecular Biology

Information flows in a single direction from your genome, i.e. DNA, to RNA (**transcription**); and from RNA to proteins (**translation**).

However, there are also feedback loops allowing information to flow the other way (**methylation**).

### Transcription

### Translation

### DNA structure and modifications

## 2. Measurement Technology

### Next Generation Sequencing (NGS)

### Applications of sequencing

## 3. Computing Technology

Computer science comprises

- Theory:
  - theoretical work on what computers can do, what kinds of problems can be computed
- Systems:
  - Computer Systems / Architecture
  - Programming Languages, Compilers
  - Operating Systems, e.g. Linux
  - Software Engineering
  - Microcontrollers and devices
- Applications:
  - things computers can be used for

**Computational thinking** is not merely knowing how to programme; but dividing a problem into tasks that can be very precisely described in order to be computable by a computer.

### Algorithms

An algorithm is a clear, step-by-step instruction of how to do something.

Some algorithms are more or less **efficient** than others.

### Memory and data structures

1 binary digit = 1 bit = 0 or 1

1 byte = 8 bits

DNA can be stored efficiently using 2 bits for each of the 4 possible bases, i.e. a four-fold compression:

A = 00, C = 01, G = 10, T = 11

### Efficiency

Travelling Salesman Problem

### Software engineering

Software engineering is all about thinking about all the different cases that your program is going to be handling or trying to think of all those cases and writing code to make sure those cases are handled:

- updating code so that it remains compatible with other software systems
- documenting computer code so that others can understand it
- testing programmes on a wide range of examples to see if they perform as expected

### Computational Biology Software

Computational biology software is what we use to transform raw data into information that you can use to make biological discoveries and guide experiments.

**Sequence alignment** refers to finding the best-matching position where a DNA sequence fits into a larger genome.

An **RNA-seq analysis** pipeline will pprocess large raw sequence files into a summary table showing which genes were present / on.

For example, a **Tuxedo** tools pipeline will thus turn reads into genes:

1. Bowtie2: fast alignment
2. TopHat2 / HISAT: spliced alignment
3. Cufflinks / StringTie: transcript assembly, quantisation
4. Cuffdiff2 / Ballgown: differential expression

## 4. Data Science Technology

### Reproducibility

Can you actually re-perform the analysis from an academic paper? Is it reproducible?

- raw data set
- tidy data set
- an explicit and exact recipe to go from raw to tidy data
- a code book describing each variable and its values

### Methods, software, analysis and applications

### The Central Dogma of Statistical Inference

Using measurements on a probabilistically selected sample to infer knowledge about a population.

### Testing

### Types of variation in genomic data

### Experimental design

### Confounding

### Power and Sample Size

### Correlation and causation

### Researcher degrees of freedom
