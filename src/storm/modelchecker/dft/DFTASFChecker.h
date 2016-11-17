#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include "src/storm/storage/dft/DFT.h"


namespace storm {
    namespace modelchecker {
        class DFTConstraint {
        public:
            virtual ~DFTConstraint() {
                
            }
            
            virtual  std::string toSmtlib2(std::vector<std::string> const& varNames) const = 0;
            virtual std::string description() const { return descript; }

            void setDescription(std::string const& descr) {
                descript = descr;
            }
            
        private:
            std::string descript;
        };
        
        class SpareAndChildPair {
            
        public:
            SpareAndChildPair(uint64_t spareIndex, uint64_t childIndex) : spareIndex(spareIndex), childIndex(childIndex) {
                
            }
            
            uint64_t spareIndex;
            uint64_t childIndex;
        };
        
        bool operator<(SpareAndChildPair const& p1, SpareAndChildPair const& p2) {
            return p1.spareIndex < p2.spareIndex || (p1.spareIndex == p2.spareIndex && p1.childIndex < p2.childIndex);
        }
        
        class DFTASFChecker {
            using ValueType = double;
            
        public:
            DFTASFChecker(storm::storage::DFT<ValueType> const&);
            void convert();
            void toFile(std::string const&);
            
        private:
            uint64_t getClaimVariableIndex(uint64_t spareIndex, uint64_t childIndex) const;
            
            storm::storage::DFT<ValueType> const& dft;
            std::vector<std::string> varNames;
            std::unordered_map<uint64_t, uint64_t> timePointVariables;
            std::vector<std::shared_ptr<DFTConstraint>> constraints;
            std::map<SpareAndChildPair, uint64_t> claimVariables;
        };
    }
}
