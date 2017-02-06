#pragma once

#include "storm-gspn/storage/gspn/GSPN.h"
#include "storm/storage/jani/Model.h"
#include "storm/storage/expressions/ExpressionManager.h"

namespace storm {
    namespace builder {
        class JaniGSPNBuilder {
        public:
            JaniGSPNBuilder(storm::gspn::GSPN const& gspn, std::shared_ptr<storm::expressions::ExpressionManager> const& expManager)
                    : gspn(gspn), expressionManager(expManager) {
                
            }
            
            virtual ~JaniGSPNBuilder() {
                // Intentionally left empty.
            }
            
            
            storm::jani::Model* build(std::string const& automatonName = "gspn_automaton");
            
            storm::jani::Variable const& getPlaceVariable(uint64_t placeId) {
                return *vars.at(placeId);
            }

            
        private:


            void addVariables(storm::jani::Model* model);

            uint64_t addLocation(storm::jani::Automaton& automaton);

            void addEdges(storm::jani::Automaton& automaton, uint64_t locId);

            const uint64_t janiVersion = 1;
            storm::gspn::GSPN const& gspn;
            std::map<uint64_t, storm::jani::Variable const*> vars;
            std::shared_ptr<storm::expressions::ExpressionManager> expressionManager;
            
        };
    }
}
