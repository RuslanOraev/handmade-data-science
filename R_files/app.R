library(shiny)
library(tidyverse)

load <- function(file = 'Titanic.csv') {
    read.csv(file)
}

get_num_col <- function(load_f){
    dat <- load_f()
    int_clmn <- sapply(dat, class) == 'integer'
    num_clmn <- sapply(dat, class) == 'numeric'
    dat.int <- dat[, int_clmn]
    dat.num <- dat[, num_clmn] 
    list(dat.int, dat.num)
}

# Application title
title_ <- titlePanel("Titanic Passengers")

panels <- 
    fluidRow(
        splitLayout(
        cellWidths = c('50%', '50%'),
        plotOutput("bar"),
        plotOutput("hist")
    )
)

btn_width <- 6

select_btns_1 <- 
    fluidRow(
        column(btn_width,
               selectInput('bar_v', 'Choose Variable',
                           choices = names(get_num_col(load)[[1]] 
                                           %>% select(-X, -PassengerId)))
        ),
        column(btn_width,
               selectInput('hist_v', 'Choose Variable',
                           choices = names(get_num_col(load)[[2]]))
        )
    )

select_btns_2 <- 
    fluidRow(
        column(btn_width,
               selectInput('bar_c', 'Choose Variable',
                           choices = colors()[seq(10, 50, 5)])
        ),
        column(btn_width,
               selectInput('hist_c', 'Choose Variable',
                           choices =  colors()[seq(60, 100, 5)])
        )
)

# Define UI for application that draws a histogram and barplot
ui <- fluidPage(
    title_, select_btns_1, select_btns_2, panels
)

# Define server logic required to draw a histogram and bar
server <- function(input, output) {
    
    output$bar <- renderPlot({
        x  <- load()
        # draw the barplot with 
        barplot(table(x[input$bar_v]),
                main = paste('Count by', input$bar_v, collapse = ''),
                xlab = input$bar_v,
                ylab = 'count',
                col = input$bar_c)
        })
    
    output$hist <- renderPlot({
        x  <- load()
        # draw the histogram
        ggplot(data = x, aes_string(input$hist_v)) +
            geom_histogram(bins = 15, 
                           fill = input$hist_c,
                           color = 'black') +# aes(get(...))
            labs(title = paste('Count by', input$hist_v, collapse = ''))
        
    })
   
}

# Run the application 
shinyApp(ui = ui, server = server)

  
