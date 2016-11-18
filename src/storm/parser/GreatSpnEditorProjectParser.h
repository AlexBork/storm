#pragma once

#include "storm-config.h"
#ifdef USE_XERCES
#include <string>

#include <xercesc/parsers/XercesDOMParser.hpp>
#include <xercesc/util/XMLString.hpp>

#include "storm/storage/gspn/GSPN.h"

#include "storm/storage/gspn/GspnBuilder.h"

namespace storm {
    namespace parser {
        class GreatSpnEditorProjectParser {
            
        public:
            
            /*!
             * Parses the given file into the GSPN.
             *
             * @param filename The name of the file to parse.
             * @return The resulting GSPN.
             */
            storm::gspn::GSPN* parse(xercesc::DOMElement const*  elementRoot);
        private:
            void traverseProjectElement(xercesc::DOMNode const* const node);
            
            void traverseGspnElement(xercesc::DOMNode const* const node);
            void traverseNodesElement(xercesc::DOMNode const* const node);
            void traverseEdgesElement(xercesc::DOMNode const* const node);
            
            void traversePlaceElement(xercesc::DOMNode const* const node);
            void traverseTransitionElement(xercesc::DOMNode const* const node);
            void traverseArcElement(xercesc::DOMNode const* const node);
            
            
            // the constructed gspn
            storm::gspn::GspnBuilder builder;

        };
    }
}
#endif
