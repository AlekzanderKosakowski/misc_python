# Equations taken from:
#   http://www.astro.uu.se/valdwiki/Air-to-vacuum%20conversion

# "For the vacuum to air conversion the formula from Donald Morton
#   (2000, ApJ. Suppl., 130, 403) is used for the refraction index,
#   which is also the IAU standard:"

def vac2air(vacuum):

  # Converts vacuum wavelengths (in Angstrom) to air wavelengths (in Angstrom)

  s = 10**4/vacuum
  n = 1 + 0.0000834254 + 0.02406147 / (130 - s**2) + 0.00015998/(38.9 - s**2)

  air = vacuum/n
  return(air)


# "The opposite conversion (air-to-vacuum) is less trivial because n depends
#   on λvac and conversion equations with sufficient precision are not
#   readily available. VALD3 tools use the following solution derived by
#   N. Piskunov:"

def air2vac(air):

  # Converts air wavelengths (in Angstrom) to vacuum wavelengths (in Angstrom)

  s = 10**4/air
  n = 1 + 0.00008336624212083 + 0.02408926869968 / (130.1065924522 - s**2) + 0.0001599740894897 / (38.92568793293 - s**2)

  vacuum = air*n
  return(vacuum)
