package Players;

import Utilities.Move;
import Utilities.PlayerStateTreeATP;
import Utilities.StateTree;

import java.util.ArrayList;
import java.util.PriorityQueue;

/**
 * This player builds a basic state tree to pick a move
 *
 * @author ATPanetta
 */

public class PlayerATP extends Player {
    double timeRatio = 0.25; // percentage of the time limit given to building the tree
    int bufferTime = 1000; // time in ms before the time limit to stop searching, if still searching
    int maxDepth = 5; // maximum depth to build the tree

    PlayerStateTreeATP currentState; // root of search tree
    int alpha; // the value of the best choice we have so far in search
    int beta; // the value of our opponents best choice so far in search
    Move nextMove; // keeps track of our result during minimax recursion
    PriorityQueue<PlayerStateTreeATP> pQueue;
    long startTime;
    boolean timer1, timer2;

    public PlayerATP(String n, int t, int l) {
        super(n, t, l);
        alpha = Integer.MIN_VALUE;
        beta = Integer.MAX_VALUE;
        nextMove = null;
        pQueue = new PriorityQueue<PlayerStateTreeATP>();
    }

    public PlayerATP(String n, int t, int l, double ratio, int buffer, int depth) {
        super(n, t, l);
        alpha = Integer.MIN_VALUE;
        beta = Integer.MAX_VALUE;
        nextMove = null;
        timeRatio = ratio;
        bufferTime = buffer;
        maxDepth = depth;
        pQueue = new PriorityQueue<PlayerStateTreeATP>();
    }

    public Move getMove(StateTree state) {
        System.out.println("Getting move...");
        startTime = System.currentTimeMillis();
        timer1 = false;
        timer2 = false;
        //Thread thread = new Thread(new Runnable() {

//            @Override
//            public void run() {
//                if (!timer1 && System.currentTimeMillis() - startTime > timeLimit * 1000 * timeRatio) {
//                    timer1 = true;
//                }
//                if (!timer2 && System.currentTimeMillis() - startTime > timeLimit * 1000 - bufferTime) {
//                    timer2 = true;
//                }
//            }
//        });

//        thread.start();
        currentState = new PlayerStateTreeATP(state);
        nextMove = null;
        pQueue.clear();
        System.out.println("Building tree...");
        buildTree(currentState);
        System.out.println("Searching tree...");
        abMax(currentState);
        return nextMove;
    }

    /* Builds the search tree of the current state */
    private void buildTree(PlayerStateTreeATP state) {
        long elapsedTime;
        for (int i = 0; i < maxDepth; i++) {
            System.out.println("i = " + i);
            elapsedTime = System.currentTimeMillis() - startTime;
            System.out.println("elapsed time = " + elapsedTime);

            if (elapsedTime > this.timeLimit * 1000 * timeRatio) break;
            buildDLS(state, i);
        }
    }

    /* Recursively builds the search tree through depth limited search */
    private void buildDLS(PlayerStateTreeATP state, int depth) {
        if (depth == 0 ||
                System.currentTimeMillis() - startTime > this.timeLimit * 1000 * timeRatio) { // at max depth or time, stop building
//                    timer1) {
            return;
        }
        ArrayList<PlayerStateTreeATP> children = state.makeChildren(); // expands the node

        for (PlayerStateTreeATP child : children) {
            if (nextMove == null) nextMove = child.getLastMove();
            child.setValue(evaluate(child));
            pQueue.add(child);
        }
        if (!pQueue.isEmpty()) buildDLS(pQueue.poll(), depth - 1);
        else return;
    }

    private int abMax(PlayerStateTreeATP state) {
 //       if (System.currentTimeMillis() - startTime > this.timeLimit * 1000 - bufferTime) {
        if (timer2) {
            if (state.getParent() == null) { // root node
                nextMove = state.getChildren().remove(0).getLastMove();
            }
            return state.getValue();
        }
        if (!state.hasChildren()) return state.getValue();
        int max = Integer.MIN_VALUE;
        int value;
        for (PlayerStateTreeATP child : state.getChildren()) {
            value = abMin(child);
            if (value > max) {
                max = value;
                if (state.getParent() == null) { // root node
                    nextMove = child.getLastMove();
                }
                if (max >= beta) return max; // pruning
                if (max > alpha) alpha = max; // just updating
            }
        }

        return max;
    }

    private int abMin(PlayerStateTreeATP state) {
        if (!state.hasChildren() ||
//                System.currentTimeMillis() - startTime > this.timeLimit * 1000 - bufferTime) { // leaf node or time limit)
        timer2) {
            return evaluate(state);
        }
        int min = Integer.MAX_VALUE;
        int value;
        for (PlayerStateTreeATP child : state.getChildren()) {
            value = abMax(child);
            if (value < min) {
                min = value;
            }
            if (min <= alpha) return min; // pruning
            if (min < beta) beta = min; // just updating
        }

        return min;
    }

    /* checking horizontal connections
     *  Returns 0 if player has a winning state
     *  Returns 1 if player is 1 away from winning
     */
    private int checkHorizontal(int player) {
        int[][] board = this.currentState.getBoardMatrix();
        int rows = board.length;
        int cols = board[0].length;
        int winNumber = this.currentState.winNumber;
        int count = 0;
        int max = 0; //max # of connections seen across the board
        int win = 0;

        // Horizontal check
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < cols; j++) {

                if (board[i][j] == player) {
                    count++;
                    if (count > max) { //store max # of connections
                        max = count;
                    }
                }
                else if (board[i][j] == 0 && count == winNumber-1) { //check if empty
                    return 1; //we can use this spot to win
                } else
                    count = 0;

                if (count >= winNumber) {
                    win = 0;
                    return win;
                } else {
                    return max;
                }
            }
        }

        return max;
    }

    /* checking vertical connections
     *  Returns 0 if player has a winning state
     *  Returns 1 if player is 1 away from winning
     */
    private int checkVertical(int player) {
        int[][] board = this.currentState.getBoardMatrix();
        int rows = board.length;
        int cols = board[0].length;
        int winNumber = this.currentState.winNumber;
        int count = 0;
        int max = 0;

        // Vertical check
        for (int i = 0; i < cols; i++)
        {
            for (int j = 0; j < rows; j++) {

                if (board[j][i] == player) {
                    count++;
                    if (count > max) {
                        max = count;
                    }
                }
                else if (board[j][i] == 0 && count == winNumber-1){ //if spot is empty
                    return 1;
                } else
                    count = 0;

                if (count >= winNumber) {
                    return 0;
                } else {
                    return max;
                }
            }
        }
        return max;
    }

    /* checking forward diagonal connections
     *  Returns 0 if player has a winning state
     *  Returns 1 if player is 1 away from winning
     */
    private int checkDiagonalForward(int player) {
        int[][] board = this.currentState.getBoardMatrix();
        int rows = board.length;
        int cols = board[0].length;
        int winNumber = this.currentState.winNumber;
        int count = 0;
        int max = 0;
        int win = 0;

        //continue checking diagonal loop
        boolean checkDiagonal = false;

        /* Row and Column Iterators
         * Incremented every time a match is found
         * Moves our place 1 over in both the row and column to do a diagonal check
         */
        int colIterator = 1;
        int rowIterator = 1;

        for(int i = 0; i < rows ; i++){
            for(int j = 0; j < cols; j++) {
                if(board[i][j] == player) { //if player token is found
                    count += 1; //iterate counter
                    checkDiagonal = true; //begin diagonal check loop
                    while(checkDiagonal){ //goes through diagonally looking for player tokens
                        if((colIterator + i <= rows - 1) && (rowIterator + j <= cols - 1)){ //move 1 right, 1 down
                            if(board[i + colIterator][j + rowIterator] == player){ //if our token is found, increment counter
                                count += 1;
                                if (count > max) {
                                    max = count;
                                }
                            } else if (board[i + colIterator][j + rowIterator] == 0 && count == winNumber-1 ) { //if spot is open
                                return 1; //there is an empty spot we can use to win
                            }
                        }
                        //adds 1 to checkers
                        colIterator += 1;
                        rowIterator += 1;
                        //check if in bounds
                        if(colIterator == rows -1 || rowIterator == cols -1){ //if OOB, exit loop
                            checkDiagonal = false;
                            break;
                        }
                        //did we win?
                        if(count >= winNumber){
                            checkDiagonal = false; //win is found, done checking
                            return 0;
                        } else {
                            return max;
                        }
                    }
                }
                //did we win?
                if(count >= winNumber){
                    return 0;
                } else {
                    //win++;
                }
                //reset iterators
                count = 0;
                colIterator = 1;
                rowIterator = 1;
            }
        }
        return max;
    }

    /* checking backward diagonal connections
     *  Returns 0 if player has a winning state
     *  Returns 1 if player is 1 away from winning
     */
    private int checkDiagonalBack(int player) {
        int[][] board = this.currentState.getBoardMatrix();
        int rows = board.length;
        int cols = board[0].length;
        int winNumber = this.currentState.winNumber;
        int count = 0;
        int max = 0;
        int win = 0;

        //continue checking diagonal loop
        boolean checkDiagonal = false;

        /* Row and Column Iterators
         * Incremented every time a match is found
         * Moves our place 1 over in both the row and column to do a diagonal check
         */
        int colIterator = 1;
        int rowIterator = 1;

        for(int i = 0; i < rows ; i++){
            for(int j = 0; j < cols; j++) {
                if (board[i][j] == player) { //if player token is found
                    count+=1; //iterate counter
                    checkDiagonal = true; //begin diagonal check loop
                    while (checkDiagonal) { //goes through diagonally looking for player tokens
                        if (i + colIterator < rows && j - rowIterator >= 0) { //move 1 left, 1 down in bounds
                            if (board[i+colIterator][j-rowIterator] == player) { //if our token is found, increment counter
                                count+=1;
                                if (count > max){
                                    max = count;
                                }
                            } else if (board[i+colIterator][j-rowIterator] == 0 && count == winNumber-1) {
                                return 1; // there is an empty spot we can use to win!
                            }
                            //win++;
                        }
                    }
                    //adds 1 to checkers
                    colIterator += 1;
                    rowIterator += 1;

                    //check if in bounds
                    if (colIterator == 0 || rowIterator == cols - 1) { //if OOB, exit loop
                        checkDiagonal = false;
                        break;
                    }

                    //did we win?
                    if(count >= winNumber){
                        checkDiagonal = false; //win is found, done checking
                        return 0;
                    } else {
                        win++;
                    }
                }
            }
            //did we win?
            if(count >= winNumber){
                return 0;
            } else {
                //win++;
            }
            //reset iterators
            count = 0;
            colIterator = 1;
            rowIterator = 1;
        }
        return max;
    }

    /* checks if contains win for player number
     *  Returns true if player has a winning state
     *  false if else
     */
    private boolean containsWin(int player){
        if (this.checkHorizontal(player) == 0 ||
                this.checkVertical(player) == 0 ||
                this.checkDiagonalBack(player) == 0 ||
                this.checkDiagonalForward(player) == 0) {
            return true;
        } else
            return false;
    }

    /* heuristic evaluation function for a single state
     * scale: -100 to 100
     * -100 = best possible state
     *  -90 = we are 1 move away from a win
     *  -60 = we have more connections
     *    0 = neutral state
     *   60 = they have more connections
     *   90 = they are 1 move away from a win
     *  100 = worst possible state
     */
    public int evaluate(PlayerStateTreeATP state) {
        int value = 0;
        int me = state.turn;
        int opp = 0;

        if (me == 1){
            opp = 2;
        } else {
            opp = 1;
        }

        /* ======= CASE 1 =======
         *  STATE CONTAINS A WIN */
        if (this.containsWin(me)) { //if I win, return 100
            value = -100;
        } else if (this.containsWin(opp)){
            value = 100;
        }

        /* ======= CASE 2 =======
         *  1 MOVE AWAY FROM WIN */
        if (this.checkHorizontal(me) == 1 || this.checkVertical(me) == 1 ||
                this.checkDiagonalBack(me) == 1 || this.checkDiagonalForward(me) == 1) {
            value = -90;
        } else if (this.checkHorizontal(opp) == 1 || this.checkVertical(opp) == 1 ||
                this.checkDiagonalBack(opp) == 1 || this.checkDiagonalForward(opp) == 1) {
            value = 90;
        }

        /* ======= CASE 3 =======
         *  NEUTRAL - NO WINS FOR EITHER */
        if ( this.checkHorizontal(me) > 1|| this.checkVertical(me) > 1 ||
                this.checkDiagonalBack(me) > 1 || this.checkDiagonalForward(me) > 1) {
            value = 0;
        } else if (this.checkHorizontal(opp) > 1 || this.checkVertical(opp) > 1 ||
                this.checkDiagonalBack(opp) > 1 || this.checkDiagonalForward(opp) > 1) {
            value = 0;
        }

        /* ======= CASE 4 =======
         *  WHO HAS MORE IN A ROW */

        int maxMe = 0;
        int maxOpp = 0;

        int h1 = this.checkHorizontal(me);
        int v1 = this.checkVertical(me);
        int db1 = this.checkDiagonalBack(me);
        int df1 = this.checkDiagonalForward(me);

        int h2 = this.checkHorizontal(opp);
        int v2 = this.checkVertical(opp);
        int db2 = this.checkDiagonalBack(opp);
        int df2 = this.checkDiagonalForward(opp);

        maxMe = Math.max(Math.max(Math.max(h1, v1), db1),df1);
        maxOpp = Math.max(Math.max(Math.max(h2, v2), db2),df2);

        if (maxMe > maxOpp) {
            value = -60;
        } else if (maxOpp > maxMe) {
            value = 60;
        }

        return value;
    }
}
