target_sources(WarpX
  PRIVATE
    ComputeDivE.cpp
    EvolveB.cpp
    EvolveBPML.cpp
    EvolveE.cpp
    EvolveEPML.cpp
    EvolveF.cpp
    EvolveFPML.cpp
    FiniteDifferenceSolver.cpp
    MacroscopicEvolveE.cpp
)

if(WARPX_MAG_LLG)
    target_sources(WarpX
      PRIVATE
        MacroscopicEvolveHM.cpp
        MacroscopicEvolveHM_2nd.cpp
    )
endif()

add_subdirectory(MacroscopicProperties)
