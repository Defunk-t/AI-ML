#include <cstdlib>
#include <iostream>
#include <limits>

typedef unsigned int GAIndex;

namespace GA
{
	class Eggholder
	{
		public:
			Eggholder(GAIndex bits, GAIndex populationSize);

		private:
//			static const GAIndex MAX_ARRAY = std::numeric_limits<GAIndex>::max();
			static const GAIndex MAX_ARRAY = 262144;
			bool generation[MAX_ARRAY]{};
			GAIndex bits, populationSize;
			void generatePopulation();
			void printPopulation();
			GAIndex getPos(GAIndex individual, GAIndex bit);
	};
}
