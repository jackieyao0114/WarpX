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
    MacroscopicEvolveM.cpp
    MacroscopicEvolveM_2nd.cpp
    MacroscopicEvolveHM.cpp
)

add_subdirectory(MacroscopicProperties)
