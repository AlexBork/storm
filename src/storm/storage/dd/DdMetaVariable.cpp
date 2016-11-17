#include "src/storm/storage/dd/DdMetaVariable.h"

#include "src/storm/utility/macros.h"

namespace storm {
    namespace dd {
        template<DdType LibraryType>
        DdMetaVariable<LibraryType>::DdMetaVariable(std::string const& name, int_fast64_t low, int_fast64_t high, std::vector<Bdd<LibraryType>> const& ddVariables) : name(name), type(MetaVariableType::Int), low(low), high(high), ddVariables(ddVariables) {
            this->createCube();
        }
        
        template<DdType LibraryType>
        DdMetaVariable<LibraryType>::DdMetaVariable(std::string const& name, std::vector<Bdd<LibraryType>> const& ddVariables) : name(name), type(MetaVariableType::Bool), low(0), high(1), ddVariables(ddVariables) {
            this->createCube();
        }
        
        template<DdType LibraryType>
        std::string const& DdMetaVariable<LibraryType>::getName() const {
            return this->name;
        }
        
        template<DdType LibraryType>
        MetaVariableType DdMetaVariable<LibraryType>::getType() const {
            return this->type;
        }
        
        template<DdType LibraryType>
        int_fast64_t DdMetaVariable<LibraryType>::getLow() const {
            return this->low;
        }
        
        template<DdType LibraryType>
        int_fast64_t DdMetaVariable<LibraryType>::getHigh() const {
            return this->high;
        }
        
        template<DdType LibraryType>
        std::size_t DdMetaVariable<LibraryType>::getNumberOfDdVariables() const {
            return this->ddVariables.size();
        }
        
        template<DdType LibraryType>
        std::vector<Bdd<LibraryType>> const& DdMetaVariable<LibraryType>::getDdVariables() const {
            return this->ddVariables;
        }
        
        template<DdType LibraryType>
        Bdd<LibraryType> const& DdMetaVariable<LibraryType>::getCube() const {
            return this->cube;
        }
        
        template<DdType LibraryType>
        void DdMetaVariable<LibraryType>::createCube() {
            STORM_LOG_ASSERT(!this->ddVariables.empty(), "The DD variables must not be empty.");
            auto it = this->ddVariables.begin();
            this->cube = *it;
            ++it;
            for (auto ite = this->ddVariables.end(); it != ite; ++it) {
                this->cube &= *it;
            }
        }
        
        template class DdMetaVariable<DdType::CUDD>;
        template class DdMetaVariable<DdType::Sylvan>;
    }
}
