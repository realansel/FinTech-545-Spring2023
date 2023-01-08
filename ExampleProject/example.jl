using StatsBase
using Printf

#Function to play the game
function playGame(guess::Int64,switch::Bool)
    #Set up the Doors and randomly pick the winning door
    doors = [0,0,0]
    winner = rand(1:3)
    doors[winner] = 1

    #Get the indexes of the losing doors
    losers = findall(x->x==0,doors)

    #Switch Guess if that is the strategy
    if switch
        #Pick the losing door to show.  Don't show the guessed door if it is a loser
        show = losers[1]
        if show == guess
            show = losers[2]
        end

        choices = [1,1,1] #narrow down the choices
        choices[show] = 0
        choices[guess] = 0

        #Pick the remaining door - switch guess
        guess = findfirst(x->x==1,choices)
    end

    #return the status of our guess
    return doors[guess]
end

println("############################################################")
#play the game 1000 times not switching
winsKeep = playGame.(fill(1,1000),false)
pctWinsKeep = mean(winsKeep)
@printf("Win Percent Not Switching: %.2f%%\n",pctWinsKeep*100)

#play the game 1000 time switching
winsSwitch = playGame.(fill(1,1000),true)
pctWinsSwitch = mean(winsSwitch)
@printf("Win Percent Switching: %.2f%%\n",pctWinsSwitch*100)

#simulaiton study - play the game 1000 times 1000 times
winsKeep = zeros(1000)
winsSwitch = zeros(1000)

for i in 1:1000
    winsKeep[i] = mean(playGame.(fill(1,1000),false))
    winsSwitch[i] = mean(playGame.(fill(1,1000),true))
end

println("############################################################")
println("")
#Descriptive Stats on the Study
println("Wins Not Switching")
describe(winsKeep)
println("SD: $(StatsBase.std(winsKeep))")


println("############################################################")
println("")
println("Wins Switching")
describe(winsSwitch)
println("SD: $(StatsBase.std(winsSwitch))")
