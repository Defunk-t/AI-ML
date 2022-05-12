#include <cstdlib>
#include <array>

#include "GeneticAlgorithm.cpp"

using std::array;
using std::string;

namespace GA
{
	typedef array<char,256> String;
	typedef short StringIndex;

	class StringEvolver: public GeneticAlgorithm<String>
	{
	  private:
		static char getRandomChar() { return (char) (32 + rand() % 95); };
		String createIndividual() const override;
		String createIndividual(String, String) const override;
		int testFitness(String) const override;
		bool terminateCondition(int fitness) const override;

		string targetString;
		StringIndex targetStringSize;

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
