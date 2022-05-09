#include <cstdlib>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

using std::string;

namespace GA
{
	class StringEvolver
	{
	  private:

		class Individual {

		  private:
			static const string VALID_CHARS;
			static const unsigned char VALID_CHARS_RANDMOD;
			constexpr static const float MUTATION_CHANCE = 0.001;
			constexpr static const float NON_MUTATION_CHANCE = (1 - MUTATION_CHANCE) / 2;
			string value;
			unsigned short fitness = 0;
			void addChar(char newChar, char targetChar);
			void randomChar(char targetChar);

		  public:
			unsigned short getFitness() const { return fitness; };
			void print() const { std::cout << value << "\tfitness: " << fitness; };
			explicit Individual(const string& targetString);
			Individual(const string& targetString, const Individual* parentA, const Individual* parentB);
		};

		static const unsigned char POPULATION_SIZE = 250;
		static const unsigned char GENE_POOL_SIZE = std::max(2, (int)(POPULATION_SIZE * 0.24));
		static const unsigned char ELITE = GENE_POOL_SIZE * 0.05;

		string targetString;
		Individual* population[POPULATION_SIZE]{};
		long generation = 0;

		void printGenerationString() const;
		void randomisePopulation();
		void printPopulation() const;
		void sort();
		bool areWeThereYet();
		bool evolve();

	  public:
		explicit StringEvolver(const string& targetString);
	};
}
