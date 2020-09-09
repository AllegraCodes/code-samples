package project5;

import java.util.ArrayList;

public class Bag {
    String name; // The name of this bag as a single lower case letter
    int capacity; // The maximum capacity of this bag in kg
    int weight; // The weight of the contents of the bag
    ArrayList<Item> contents; // List of Items assigned to this bag

    public Bag(String name, int capacity) {
        this.name = name;
        this.capacity = capacity;
        this.weight = 0;
        this.contents = new ArrayList();
    }

    public Bag(String name, String capacity) {
        this.name = name;
        this.capacity = Integer.parseInt(capacity);
        this.weight = 0;
        this.contents = new ArrayList();
    }

    public void addItem(Item i) {
        this.contents.add(i);
        i.isAssigned = true;
        this.weight += i.weight;
    }

    public void removeItem(Item i) {
        this.contents.remove(i);
        i.isAssigned = false;
        this.weight -= i.weight;
    }

    public String getName(){
        return this.name;
    }

    public int getCapacity() {
        return this.capacity;
    }

    public ArrayList<Item> getContents() {
        return contents;
    }

    public int getWeight() {
        return weight;
    }
}
