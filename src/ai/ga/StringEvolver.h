#include <array>
#include <cstdlib>
#include <iostream>

#include "GeneticAlgorithm.cpp"

using std::array;
using std::cout;
using std::endl;
using std::string;

namespace GA
{
	typedef array<char,256> String;
	typedef short StringIndex;

	class StringEvolver: public GeneticAlgorithm<String>
	{
	  private:
		string targetString;
		StringIndex targetStringSize;

		static char getRandomChar();

		String createIndividual() const override;
		String createIndividual(String, String) const override;
		int testFitness(String) const override;
		bool terminateCondition(int fitness) const override;
		void print() const override;

	  public:
		explicit StringEvolver(
			GA_Int populationSize,
			GA_Int genePoolSize,
			GA_Int eliteCount,
			double mutationChance,
			const string &targetString
		);
	};
};
