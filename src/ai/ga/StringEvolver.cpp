#include "StringEvolver.h"

namespace GA
{
	// Algorithm

	StringEvolver::StringEvolver(const string& targetString)
	{
		this->targetString = targetString;
		randomisePopulation();
		while (!evolve()) if (generation % 10000 == 0) printPopulation();
		const double effort = (double)generation * POPULATION_SIZE;
		std::cout << "String found.\tEffort: " << effort << " | " << std::pow(26, targetString.size()) / 2 / effort << "x brute-force speed";
	};

	void StringEvolver::randomisePopulation()
	{
		for (auto & i : population)
			i = new Individual(targetString);
	};

	void StringEvolver::printGenerationString() const
	{
		std::cout << "Generation: ";
		if (generation < 1000) std::cout << generation;
		else if (generation < 1000000) std::cout << (double)generation / 1000 << "K";
		else std::cout << (double)generation / 1000000 << "M";
	};

	void StringEvolver::printPopulation() const
	{
		printGenerationString();
		std::cout << std::endl;
		for (unsigned char i = 0; i < ELITE; i++)
		{
			std::cout << "\t Elite[" << (int)i + 1 << "] ";
			population[i]->print();
			std::cout << std::endl;
		}
	};

	void StringEvolver::sort()
	{
		for (unsigned char i = 0; i < POPULATION_SIZE - 1; i++)
			for (unsigned char j = 1; j < POPULATION_SIZE - i; j++)
				if (population[j]->getFitness() < population[j - 1]->getFitness())
				{
					Individual* temp = population[j];
					population[j] = population[j - 1];
					population[j - 1] = temp;
				}
	};

	bool StringEvolver::areWeThereYet()
	{
		sort();
		if (population[0]->getFitness() == 0) return true;
		else return false;
	};


	bool StringEvolver::evolve()
	{
		// Establish the gene pool
		Individual* genePool[GENE_POOL_SIZE];
		for (unsigned short i = 0; i < POPULATION_SIZE; i++)
		{
			// The best fitting individuals go in the gene pool
			if (i < GENE_POOL_SIZE) genePool[i] = population[i];
			// The rest are deleted to avoid memory leak
			else delete population[i];
		}

		++generation;

		// Replace non-elite individuals
		for (unsigned short i = ELITE; i < POPULATION_SIZE; i++)
		{
			// Select random parents from gene pool
			// no asexual
			unsigned short r1 = rand() % GENE_POOL_SIZE, r2;
			do r2 = rand() % GENE_POOL_SIZE;
			while (r1 == r2);

			// Mate to create new individual
			population[i] = new Individual(targetString, population[r1], population[r2]);
		}

		// Delete non elites from gene pool to avoid memory leak
		for (unsigned short i = ELITE; i < GENE_POOL_SIZE; i++)
			delete genePool[i];

		return areWeThereYet();
	};

	// Individual

	const string StringEvolver::Individual::VALID_CHARS = "abcdefghijklmnopqrstuvwxyz ";

	const unsigned char StringEvolver::Individual::VALID_CHARS_RANDMOD = VALID_CHARS.size() - 1;

	void StringEvolver::Individual::addChar(char newChar, char targetChar)
	{
		value += newChar;
		if (newChar != targetChar) ++fitness;
	};

	void StringEvolver::Individual::randomChar(char targetChar)
	{
		addChar(VALID_CHARS[rand() % VALID_CHARS_RANDMOD], targetChar);
	}

	StringEvolver::Individual::Individual(const string& targetString)
	{
		for (char i : targetString)
			randomChar(i);
	};

	StringEvolver::Individual::Individual(const string& targetString, const Individual* parentA, const Individual* parentB)
	{
		for (unsigned int i = 0; i < targetString.size(); i++)
		{
			const float random = rand() % 10000 / 10000;
			if (random < MUTATION_CHANCE)
				randomChar(targetString[i]);
			else if (random < 1 - NON_MUTATION_CHANCE)
				addChar(parentA->value[i], targetString[i]);
			else
				addChar(parentB->value[i], targetString[i]);
		}
	};
}
